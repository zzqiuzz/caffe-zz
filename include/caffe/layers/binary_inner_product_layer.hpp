#ifndef CAFFE_BINARY_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_BINARY_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BinaryInnerProductLayer : public Layer<Dtype> {
 public:
	 explicit BinaryInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BinaryInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  /*user's function*/
  void cpuMeanClampBinarizeParam(const shared_ptr<Blob<Dtype> > weights, const shared_ptr<Blob<Dtype> > wb);
  void gpuMeanClampBinarizeParam(const shared_ptr<Blob<Dtype> > weights, const shared_ptr<Blob<Dtype> > wb);
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
  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights

  shared_ptr<Blob<Dtype> > W_b;
  shared_ptr<Blob<Dtype> > W_buffer;
  Blob<Dtype> filterMean;
  Blob<Dtype> Alpha;
  //Blob<Dtype> sumHelper;//used for vector sum
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
