#ifndef CAFFE_BINARY_CONV_LAYER_HPP_
#define CAFFE_BINARY_CONV_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h" 
//#include "caffe/common.hpp"
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
	//W_b store binarized weights
	//W_buffer store original weights.
	shared_ptr<Blob<Dtype> > W_b;
	shared_ptr<Blob<Dtype> > W_buffer;
	Blob<Dtype> alphas_; 
	vector<Dtype> filterMean;
	vector<Dtype> Alpha;
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
	
	void cpuMeanClampBinarizeConvParam(const shared_ptr<Blob<Dtype> > weights, const shared_ptr<Blob<Dtype> > wb);
	void gpuMeanClampBinarizeConvParam(const shared_ptr<Blob<Dtype> > weights, const shared_ptr<Blob<Dtype> > wb);
	inline void copyCpuFromTo(const shared_ptr<Blob<Dtype> > orig, const shared_ptr<Blob<Dtype> > buf){
		CHECK_EQ(orig->count(), buf->count());
		caffe_copy(orig->count(), orig->cpu_data(), buf->mutable_cpu_data());
	}
	inline void copyGpuFromTo(const shared_ptr<Blob<Dtype> > orig, const shared_ptr<Blob<Dtype> > buf){
		CHECK_EQ(orig->count(), buf->count());
		caffe_copy(orig->count(), orig->gpu_data(), buf->mutable_gpu_data());
	}
	inline Dtype clampWeights(Dtype w){
		return w = (w) < -1 ? -1 : (w) >1 ? 1 : (w);
	}
	inline Dtype signWeights(Dtype w){
		return w = (w) >= 0 ? 1 : -1;
	}



};

}


#endif