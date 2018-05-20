#include <vector>
#include "caffe/layers/binary_conv_layer.hpp"

namespace caffe{

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

//Initialize binaried weights
template BinaryConvolutionLayer<Dtype>::binaryConvInit(){
	W_b = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
	W_buffer = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
	W_b->ReshapeLike(this->blob_[0]);
	W_buffer->ReshapeLike(this->blob_[0]);
}
//blob[0]-->W_b
template BinaryConvolutionLayer<Dtype>::binarizeFloatWeights(const shared_ptr<Blob<Dtype> > weights,
			const shared_ptr<Blob<Dtype> > wb){
	
	
}

template BinaryConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top){
	//TODO:
	//convert float weights to bianry 
	binarizeFloatWeights(this->blob_[0],W_b);
	//store float weights to W_buffer
	copyFromTo(this->blob_[0],W_buffer);
	//reinitialize blob_ with binarized weights W_b.
	copyFromTo(W_b,this->blob_[0]); 
	//normal conv operations,directly copied from conv_layer.cpp
	const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template BinaryConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top){


}

#ifdef CPU_ONLY
STUP_GPU(BinaryConvolutionLayer);
#endif

INSTANTIATE_CLASS(BinaryConvolution);

}