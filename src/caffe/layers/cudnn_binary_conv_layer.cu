#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_binary_conv_layer.hpp"

namespace caffe {

__global__ void sync_binary_conv_groups() { }
#define sign(x) ((x)>=0?1:-1)
#define clamp(x) ((x) < -1 ? -1 : (x) >1 ? 1 : (x))
template <typename Dtype>
__global__ void CudnnBinaryGpu_binarize(const int num, const int weight_col, const Dtype* alpha,const Dtype* in, Dtype* out){
	CUDA_KERNEL_LOOP(index, num){
		int n = index / weight_col; 
		const Dtype binarycode = in[index] >= 0 ? 1 : -1; 
		out[index] = binarycode*alpha[n];

		/*for (int coor = 0; coor < weight_col; coor++){
			out[index*weight_col + coor] = sign(in[index*weight_col + coor]) * alpha[index];
		}*/
	}
}
template <typename Dtype>
__global__ void Gradient_adder(const int num,const int weight_dim,const Dtype* weight,Dtype* weight_diff,const Dtype* alpha){
	CUDA_KERNEL_LOOP(index, num){ 
		const int n = index / weight_dim;
		Dtype multiplier = 0;
		if (abs(weight[index]) <= 1)
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
__global__ void Mean_sub(const int num,const int weight_dim,const Dtype* mean_data,Dtype* out){
	CUDA_KERNEL_LOOP(index,num){
		int n = index / weight_dim ;
		out[index]-=mean_data[n];
	}
}
template <typename Dtype>
void CuDNNBinaryConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	Phase phase = this->layer_param_.phase();
	//const int num = this->blobs_[0]->num();
	const int N = this->blobs_[0]->count();
	const int num = this->num_output_;
	const int div = N / num;
	
	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* binaryweight = this->W_b.mutable_gpu_data();
	//caffe_copy<Dtype>(N, weight, binaryweight);
	caffe_gpu_abs(N,weight,binaryweight);
	if(this->layer_param_.debug_param().xnorno_grad() && phase == TRAIN){//only in train phase! 
		//calculate mean_.
		caffe_gpu_gemv<Dtype>(CblasNoTrans, num, div, 1. / div, weight, this->weight_sum_multiplier.gpu_data(), 0.,
			this->mean_.mutable_gpu_data()); 
		//extract mean.
		const Dtype* mean_data=this->mean_.gpu_data();
		Mean_sub<Dtype><< <CAFFE_GET_BLOCKS(N),CAFFE_CUDA_NUM_THREADS>> >(N,div,mean_data,this->blobs_[0]->mutable_gpu_data());
		//TODO：to gpu
		/*for(int i=0;i<num;++i){
			caffe_gpu_add_scalar<Dtype>(div, -*(mean_data + i), this->blobs_[0]->mutable_gpu_data() + i*div);
		}*/
		caffe_gpu_abs(N,weight,binaryweight);
		//clamp weights
		this->blobs_[0]->clip_data(); 
	}
	
	//calculate alphas_.
	/*for (int n = 0; n < num; n++){
		caffe_gpu_asum<Dtype>(div, weight + n*div, alphas_.mutable_cpu_data() + n); 
		alphas_.mutable_cpu_data()[n] /= div; 
	}*/
	caffe_gpu_gemv<Dtype>(CblasNoTrans, num, div, 1. / div, binaryweight, this->weight_sum_multiplier.gpu_data(), 0.,
		this->alphas_.mutable_gpu_data());

	//binarize weights.
	CudnnBinaryGpu_binarize<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (
		N, div, this->alphas_.gpu_data(), weight, binaryweight);
	if(this->layer_param_.debug_param().binary_relax()){
		if(phase == TRAIN){
		//case 1: vectorize all filters in one layer
		Dtype beta=0.001;//0.001->0.01->0.05->0.1
		caffe_gpu_axpby(N,beta,weight,1-beta,binaryweight);
		
		//case 2: vectorize one filter in one layer
		/*Dtype beta = 0.001;
		for(int i = 0; i < num; i++){
		 	caffe_gpu_axpby(div,beta,weight + i * div,1-beta,binaryweight + i * div);
		}*/
		
	}
	
	}
	//normal conv operations,directly copied from conv_layer.cpp
	//const Dtype* weight = this->blobs_[0]->gpu_data();
	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* top_data = top[i]->mutable_gpu_data();
		for (int g = 0; g < this->group_; g++) {
			CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
				cudnn::dataType<Dtype>::one,
				bottom_descs_[i], bottom_data + bottom_offset_ * g,
				filter_desc_, binaryweight + this->weight_offset_ * g,
				conv_descs_[i],
				fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
				cudnn::dataType<Dtype>::zero,
				top_descs_[i], top_data + top_offset_ * g));	

			if (this->bias_term_) {
				const Dtype* bias_data = this->blobs_[1]->gpu_data();
				CUDNN_CHECK(cudnnAddTensor(handle_[g],
				  cudnn::dataType<Dtype>::one,
					bias_desc_, bias_data + bias_offset_ * g,
					cudnn::dataType<Dtype>::one,
					top_descs_[i], top_data + top_offset_ * g));
			}
		}
	}
}

template <typename Dtype>
void CuDNNBinaryConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	//const Dtype* weight = this->blobs_[0]->gpu_data();
	const Dtype* binaryweight = NULL;
	Dtype* weight_diff = NULL;
	if (this->param_propagate_down_[0]) {
		binaryweight = this->W_b.gpu_data();
		weight_diff = this->blobs_[0]->mutable_gpu_diff();
	}
	Dtype* bias_diff = NULL;
	if (this->bias_term_ && this->param_propagate_down_[1]) {
		bias_diff = this->blobs_[1]->mutable_gpu_diff();
	}
	for (int i = 0; i < top.size(); ++i) {
	  const Dtype* top_diff = top[i]->gpu_diff();
   for (int g = 0; g < this->group_; g++) {
		// Bias gradient, if necessary.
		if (this->bias_term_ && this->param_propagate_down_[1]) {
       CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g)); 
		}
   // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
              
			if(this->layer_param_.debug_param().xnorno_grad()){
				const Dtype* weight = this->blobs_[0]->gpu_data();
				const int weight_dim = this->blobs_[0]->count() / this->blobs_[0]->num();
				const int n = this->blobs_[0]->count(); 
				Gradient_adder<Dtype> << <CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS >> >
					(n, weight_dim, weight, weight_diff, this->alphas_.gpu_data()); 
			}
      }
      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (binaryweight == NULL) {
          binaryweight = this->W_b.gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, binaryweight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      

    }
   
	}
 
   sync_binary_conv_groups<<<1, 1>>>();
 }

}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBinaryConvolutionLayer);

}  // namespace caffe
#endif