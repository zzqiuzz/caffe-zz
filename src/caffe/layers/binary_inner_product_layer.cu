#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/binary_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#define sign(x) ((x)>=0?1:-1)
#define clamp(x) ((x) < -1 ? -1 : (x) >1 ? 1 : (x))

template <typename Dtype>
__global__ void binarize_kernel(const Dtype* alpha, const Dtype* in, Dtype* out, const int num, const int weight_col){
	CUDA_KERNEL_LOOP(index, num){
		int n = index / weight_col;
		const Dtype binarycode = (in[index]) >= 0 ? 1 : -1; 
		out[index] = binarycode*alpha[n];
		/*for (int coor = 0; coor < weight_col; coor++){
			out[index*weight_col + coor] = sign(in[index*weight_col + coor]) * alpha[index];
		}*/
	}
}
template <typename Dtype>
__global__ void Gradient_adder(const int num, const int weight_dim, const Dtype* weight, Dtype* weight_diff, const Dtype* alpha){
	CUDA_KERNEL_LOOP(index, num){
		const int n = index / weight_dim;
		Dtype multiplier = 0;
		if (abs(weight[index]) >= 1)
			multiplier = 0;
		else
		{
			multiplier = 1;
			multiplier *= alpha[n];
		}
		multiplier += Dtype(1) / weight_dim;
    multiplier *= (1 - 1./weight_dim);
		multiplier *= weight_dim;
		weight_diff[index] *= multiplier;
	}
}
 
template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	
  Phase phase = this->layer_param_.phase();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data(); 
  //TODO:
  //convert float weights to binary 
  const int num = this->N_;//numbers of output
  const int div = this->K_;
  int N = num*div;
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* binaryweight = W_b.mutable_gpu_data();
  caffe_copy<Dtype>(N, weight, binaryweight);
  if(this->layer_param_.debug_param().xnorno_grad()){
    //calculate mean_.
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num, div, 1. / div, weight, weight_sum_multiplier.gpu_data(), 0.,
       mean_.mutable_gpu_data()); 
    //extract mean.
     for(int i=0;i<num;++i){
      caffe_gpu_add_scalar<Dtype>(div, -*(mean_.cpu_data() + i), this->blobs_[0]->mutable_gpu_data() + i*div);
   }
    //clamp weights
     this->blobs_[0]->clip_data();
  }
  
  //calculate alphas_
  for (int n = 0; n < num; n++){
	  caffe_gpu_asum<Dtype>(div, weight + n*div, alphas_.mutable_cpu_data() + n);
	  alphas_.mutable_cpu_data()[n] /= div;
  }
  //binarize weights.
  binarize_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(alphas_.gpu_data(), weight,
	  binaryweight, N, div); 
  if(this->layer_param_.debug_param().binary_relax())  
  {
  	if(phase == TRAIN){
		//case 1: vectorize all filters in one layer
		Dtype beta=0.05;//0.001->0.01->0.05->0.1
		caffe_gpu_axpby(N,beta,weight,1-beta,binaryweight);
		
		//case 2: vectorize one filter in one layer
		/*Dtype beta = 0.001;
		for(int i = 0; i < num; i++){
		 	caffe_gpu_axpby(div,beta,weight + i * div,1-beta,binaryweight + i * div);
		}*/
	}
  }
  if (M_ == 1) {
	  caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
		  binaryweight, bottom_data, (Dtype)0., top_data);
	  if (bias_term_)
		  caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
		  this->blobs_[1]->gpu_data(), top_data);
  }
  else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
						  bottom_data, binaryweight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	const Dtype* binaryweight = W_b.gpu_data();
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
	  
	  //
    if(this->layer_param_.debug_param().xnorno_grad()){
      const Dtype* weight = this->blobs_[0]->gpu_data();
      const int weight_dim = this->blobs_[0]->count() / this->blobs_[0]->num();
      const int n = this->blobs_[0]->count();

      Gradient_adder<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >
        (n, weight_dim, weight, this->blobs_[0]->mutable_gpu_diff(), alphas_.gpu_data());
    }
	  
      
	  //
	  
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
		  (Dtype)1., top_diff, binaryweight,
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
		  (Dtype)1., top_diff, binaryweight,
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
 
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryInnerProductLayer);

}  // namespace caffe
