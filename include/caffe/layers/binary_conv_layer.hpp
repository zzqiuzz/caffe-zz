#ifndef CAFFE_BINARY_CONV_LAYER_HPP_
#define CAFFE_BINARY_CONV_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h" 
#include "caffe/common.hpp"
namespace caffe{
template <typename Dtype>
class BinaryConvolutionLayer : public BaseConvolutionLayer<Dtype>{
public:
	explicit BinaryConvolutionLayer(const LayerParameter& param)
	: BaseConvolutionLayer<Dtype>(param){}

	virtual inline const char* type() const {return "BinaryConvolution";}

protected:
	//W_b store binarized weights
	//W_buffer store original weights.
	shared_ptr<Blob<Dtype> > W_b;
	shared_ptr<Blob<Dtype> > W_buffer;

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom);
	virtual void compute_output_shape();
	/*user' function e.g binary/hardsigmoid*/
	virtual void binaryConvInit();
	virtual void binarizeFloatWeights();
	virtual void storeFloatWeights();
	virtual void reInitBlob();

}

}


#endif