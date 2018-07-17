#ifndef CAFFE_BINARY_CONV_LAYER_HPP_
#define CAFFE_BINARY_CONV_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"  
#include "caffe/layers/base_conv_layer.hpp"
namespace caffe{
template <typename Dtype>
class BinaryConvolutionLayer : public BaseConvolutionLayer<Dtype> {
public:
	explicit BinaryConvolutionLayer(const LayerParameter& param)
		: BaseConvolutionLayer<Dtype>(param){}

	virtual inline const char* type() const { return "BinaryConvolution"; }
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
protected:
	
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void compute_output_shape();
	virtual inline bool reverse_dimensions() { return false; }
	/*user' function e.g binary/hardsigmoid*/
	//W_b store binarized weights
	//W_buffer store original weights.
	Blob<Dtype> W_b;
	Blob<Dtype> alphas_;
	Blob<Dtype> mean_;
	Blob<Dtype> weight_sum_multiplier;
	inline Dtype clampWeights(Dtype w){
		return w = (w) < -1 ? -1 : (w) >1 ? 1 : (w);
	}
	inline Dtype signWeights(Dtype w){
		return w = (w) >= 0 ? 1 : -1;
	}



};

}


#endif