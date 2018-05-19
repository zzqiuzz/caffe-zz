#include <algorithm>
#include <vector>

#include "caffe/layers/all_pass_layer.hpp"

#include <iostream>
using namespace std;
#define DEBUG_AP(str) cout << str << endl;
namespace caffe{

template <typename Dtype>
void AllPassLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	for(int i = 0;i < count; ++i){
		top_data[i]=bottom_data[i];
	}
	DEBUG_AP("Here is ALl Pass Layer,forwarding.");
	DEBUG_AP(this->layer_param_.all_pass_param().key());

}
template<typename Dtype>
void AllPassLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom){
	if(propagate_down[0]){
		//const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const int count = bottom[0]->count();
		for(int i = 0;i < count;++i){
			bottom_diff[i] = top_diff[i];
		}

	}
	DEBUG_AP("Here is All Pass  Layer,backwarding.");
	DEBUG_AP(this->layer_param_.all_pass_param().key());



}
#ifdef CPU_ONLY	
STUB_GPU(AllPassLayer);
#endif

INSTANTIATE_CLASS(AllPassLayer);
REGISTER_LAYER_CLASS(AllPass);
}//namspace caffe