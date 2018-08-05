#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/ternary_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
 

template <typename Dtype>
__global__ void ternarize_data(const int num,const Dtype delta, const Dtype* in, Dtype* out){
	CUDA_KERNEL_LOOP(index, num){
		const Dtype data = in[index];
		out[index] = (data>delta) - (data<-delta);
	}
}

template <typename Dtype>
void TernaryInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	
	const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data(); 
	
	Dtype delta_scaler = 0.7;
	const int N = this->blobs_[0]->count();
	Dtype* ternaryweight = this->W_t.mutable_gpu_data();
	this->delta_ = delta_scaler * this->blobs_[0]->asum_data() / N;
	caffe_copy<Dtype>(N, this->blobs_[0]->gpu_data(), ternaryweight);
	ternarize_data<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(N,this->delta_,this->blobs_[0]->gpu_data(),
		ternaryweight);
	Dtype* temp1 = new Dtype(0);
	caffe_gpu_dot(N, this->W_t.gpu_data(), this->blobs_[0]->gpu_data(), temp1);
	Dtype* temp2 = new Dtype(0);
	caffe_gpu_dot(N, this->W_t.gpu_data(), this->W_t.gpu_data(), temp2); 
	this->alpha_ = (*temp1) / ((*temp2) + 1e-6);  
	caffe_gpu_scale(N, this->alpha_, this->W_t.gpu_data(), ternaryweight);
	
	delete temp1;
	delete temp2;
  if (M_ == 1) {
	  caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
		  ternaryweight, bottom_data, (Dtype)0., top_data);
	  if (bias_term_)
		  caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
		  this->blobs_[1]->gpu_data(), top_data);
  }
  else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
						  bottom_data, ternaryweight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void TernaryInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	const Dtype* ternaryweight = this->W_t.gpu_data();
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
		  (Dtype)1., top_diff, ternaryweight,
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
		  (Dtype)1., top_diff, ternaryweight,
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
 
}

INSTANTIATE_LAYER_GPU_FUNCS(TernaryInnerProductLayer);

}  // namespace caffe
