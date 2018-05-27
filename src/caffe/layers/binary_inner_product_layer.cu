#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/binary_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#define sign(x) ((x)>=0?1:-1)
#define clamp(x) ((x) < -1 ? -1 : (x) >1 ? 1 : (x))

template <typename Dtype>
__global__ void binarize_kernel(const Dtype* in, Dtype* out, const int num, const int kel){
	CUDA_KERNEL_LOOP(index, num){ 
		Dtype sum = 0;         
		Dtype mean = 0;
		for (int coor = 0; coor < kel; coor++){
			sum += std::abs(in[index*kel + coor]) / Dtype(kel);
			mean += in[index*kel + coor] / Dtype(kel);
		}
		for (int coor = 0; coor < kel; coor++){
			out[index*kel + coor] = sign(clamp(in[index*kel + coor] - mean))*sum;
		}
	}
}
template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::gpuMeanClampBinarizeParam(const shared_ptr<Blob<Dtype> > weights,
	const shared_ptr<Blob<Dtype> > wb){
	const int num = this->N_;//numbers of output
	const int kel = this->K_;
	int N = num*kel;
	binarize_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(weights->gpu_data(), wb->mutable_gpu_data(),num, kel); 
}
template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();

  //TODO:
  //convert float weights to binary 
  gpuMeanClampBinarizeParam(this->blobs_[0], W_b);
  /*const Dtype* cpu_d = this->blobs_[0]->cpu_data();
  const Dtype* gpu_d = this->blobs_[0]->gpu_data();
  for (int i = 0; i < 5; i++){
  std::cout << "cpu data: " << cpu_d[i];
  std::cout << " gpu data:" << gpu_d[i] << std::endl;
  }*/
  //store float weights to W_buffer
  copyGpuFromTo(this->blobs_[0], W_buffer);
  //reinitialize blob_ with binarized weights W_b.
  copyGpuFromTo(W_b, this->blobs_[0]);

  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
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
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryInnerProductLayer);

}  // namespace caffe
