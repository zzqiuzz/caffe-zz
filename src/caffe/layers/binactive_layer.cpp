#include <vector>
#include "caffe/layers/binactive_layer.hpp"

namespace caffe{
/*
template <typename Dtype>
void BinActiveLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
    
}
template <typename Dtype>
void PReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
}*/

template <typename Dtype>
void BinActiveLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
  NOT_IMPLEMENTED;
}


template <typename Dtype>
void BinActiveLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& top){
  NOT_IMPLEMENTED;


}

#ifdef CPU_ONLY
STUB_GPU(BinActiveLayer);
#endif

INSTANTIATE_CLASS(BinActiveLayer);
REGISTER_LAYER_CLASS(BinActive);





}


