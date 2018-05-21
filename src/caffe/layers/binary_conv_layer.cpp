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
	customConvInit();
}

	//Initialize binaried weights
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::customConvInit(){
	int num = this->blobs_[0]->num();
	filterMean.clear();
	Alpha.clear();
	for (int i = 0; i < num; i++)
	{
		filterMean.push_back(0);
		Alpha.push_back(0);
	}
	
	W_b = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
	W_buffer = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
	W_b->ReshapeLike(*(this->blobs_[0]));
	W_buffer->ReshapeLike(*(this->blobs_[0]));
}
template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::meanClampBinarizeConvParam(const shared_ptr<Blob<Dtype> > weights,
	const shared_ptr<Blob<Dtype> > wb){ 
	int weightsNum = weights->count();
	int num = weights->num();
	int channel = weights->channels();
	int height = weights->height();
	int width = weights->width();
	const int div = weightsNum / num;
	for (int n = 0; n < num; n++){
		for (int c = 0; c < channel; c++){
			for (int h = 0; h < height; h++){
				for (int w = 0; w < width; w++){ 
					filterMean[n] += weights->data_at(n, c, h, w);
					Alpha[n] += std::abs(weights->data_at(n, c, h, w)) / Dtype(div); 
				}
			}
		}
	} 
	for (int id = 0; id < weightsNum; id++){
		const int num = id / div;//for each filter in the layer. 
		wb->mutable_cpu_data()[id] = Alpha[num] * signWeights(clampWeights(weights->cpu_data()[id] - filterMean[num]));
	}

}


template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	//TODO:
	//convert float weights to bianry 
	meanClampBinarizeConvParam(this->blobs_[0], W_b);
	//store float weights to W_buffer
	copyFromTo(this->blobs_[0], W_buffer);
	//reinitialize blob_ with binarized weights W_b.
	copyFromTo(W_b, this->blobs_[0]);
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

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& top){


}

	/*#ifdef CPU_ONLY
	STUB_GPU(BinaryConvolutionLayer);
	#endif*/

	INSTANTIATE_CLASS(BinaryConvolutionLayer); 
	REGISTER_LAYER_CLASS(BinaryConvolution);
}