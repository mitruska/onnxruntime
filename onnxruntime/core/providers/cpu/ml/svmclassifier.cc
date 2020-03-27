// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/ml/svmclassifier.h"

namespace onnxruntime {
namespace ml {

#define ADD_IN_TYPE_SVM_CLASSIFIER_OP(in_type)                                                                                                                                                    \
  ONNX_CPU_OPERATOR_TYPED_ML_KERNEL(                                                                                                                                                              \
      SVMClassifier,                                                                                                                                                                              \
      1,                                                                                                                                                                                          \
      in_type,                                                                                                                                                                                    \
      KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<in_type>()).TypeConstraint("T2", {DataTypeImpl::GetTensorType<int64_t>(), DataTypeImpl::GetTensorType<std::string>()}), \
      SVMClassifier<in_type>);

ADD_IN_TYPE_SVM_CLASSIFIER_OP(float);
ADD_IN_TYPE_SVM_CLASSIFIER_OP(double);
ADD_IN_TYPE_SVM_CLASSIFIER_OP(int64_t);
ADD_IN_TYPE_SVM_CLASSIFIER_OP(int32_t);

template <typename T>
SVMClassifier<T>::SVMClassifier(const OpKernelInfo& info)
    : OpKernel(info),
      SVMCommon<T>(info),
      vectors_per_class_(info.GetAttrsOrDefault<int64_t>("vectors_per_class")),
      proba_(info.GetAttrsOrDefault<float>("prob_a")),
      probb_(info.GetAttrsOrDefault<float>("prob_b")),
      support_vectors_(info.GetAttrsOrDefault<float>("support_vectors")),
      post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))) {
  ORT_ENFORCE(info.GetAttrs<float>("rho", rho_).IsOK());
  ORT_ENFORCE(info.GetAttrs<float>("coefficients", coefficients_).IsOK());

  // prob_a and prob_b are optional for Z output
  ORT_ENFORCE(proba_.size() == probb_.size());

  // one of these should be valid
  ORT_ENFORCE(info.GetAttrs<std::string>("classlabels_strings", classlabels_strings_).IsOK() ||
              info.GetAttrs<int64_t>("classlabels_ints", classlabels_ints_).IsOK());

  vector_count_ = 0;
  feature_count_ = 0;
  class_count_ = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(vectors_per_class_.size()); i++) {
    starting_vector_.push_back(vector_count_);
    vector_count_ += vectors_per_class_[i];
  }

  using_strings_ = false;
  if (classlabels_strings_.size() > 0) {
    using_strings_ = true;
    class_count_ = classlabels_strings_.size();
  } else if (classlabels_ints_.size() > 0) {
    class_count_ = classlabels_ints_.size();
  } else {
    class_count_ = 1;
  }

  if (vector_count_ > 0) {
    feature_count_ = support_vectors_.size() / vector_count_;  //length of each support vector
    mode_ = SVM_TYPE::SVM_SVC;
  } else {
    feature_count_ = coefficients_.size() / class_count_;  //liblinear mode
    mode_ = SVM_TYPE::SVM_LINEAR;
    set_kernel_type(KERNEL::LINEAR);
  }

  ORT_ENFORCE(classlabels_strings_.size() > 0 || classlabels_ints_.size() > 0);
  ORT_ENFORCE(proba_.size() == probb_.size());
  ORT_ENFORCE(coefficients_.size() > 0);
  weights_are_all_positive_ = true;
  for (int64_t i = 0; i < static_cast<int64_t>(coefficients_.size()); i++) {
    if (coefficients_[i] < 0) {
      weights_are_all_positive_ = false;
      break;
    }
  }
}

template <typename LabelType>
int _set_score_svm(Tensor* Y, float max_weight, const int64_t maxclass, const int64_t n,
                   POST_EVAL_TRANSFORM post_transform_, const std::vector<float>& proba_, bool weights_are_all_positive_,
                   const std::vector<LabelType>& classlabels, LabelType posclass, LabelType negclass) {
  int write_additional_scores = -1;
  auto output_data = Y->template MutableData<LabelType>();
  if (classlabels.size() == 2) {
    write_additional_scores = post_transform_ == POST_EVAL_TRANSFORM::NONE ? 2 : 0;
    if (proba_.size() == 0) {
      if (weights_are_all_positive_ && max_weight >= 0.5)
        output_data[n] = classlabels[1];
      else if (max_weight > 0 && !weights_are_all_positive_)
        output_data[n] = classlabels[1];
      else
        output_data[n] = classlabels[maxclass];
    } else {
      output_data[n] = classlabels[maxclass];
    }
  } else if (max_weight > 0) {
    output_data[n] = posclass;
  } else {
    output_data[n] = negclass;
  }
  return write_additional_scores;
}
template <typename T>
Status SVMClassifier<T>::Compute(OpKernelContext* ctx) const {
  ORT_NOT_IMPLEMENTED();
}

template <>
Status SVMClassifier<float>::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);

  int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];

  Tensor* Y = ctx->Output(0, TensorShape({N}));

  concurrency::ThreadPool* threadpool = ctx->GetOperatorThreadPool();

  // X: [num_batches, feature_count_] where features could be coefficients or support vectors
  // support_vectors_ : [vector_count_, feature_count_]
  // vectors per class: entries that sum to vector_count_

  // Y: [num_batches, 1]

  // Total number of classifiers comparing pairs between the classes
  // e.g. if you have A, B C and D classes, the number of classifiers to compare between each pair is 6
  //      with AB, AC, AD, BC, BD and CD
  const int64_t class_count_squared = class_count_ * class_count_;
  const int64_t num_classifiers = class_count_ * (class_count_ - 1) / 2;  // == (class_count_-1)!
  const bool have_proba = proba_.size() > 0;

  int64_t nb_columns = class_count_;

  if (!have_proba && mode_ == SVM_TYPE::SVM_SVC) {
    if (class_count_ > 2)
      nb_columns = num_classifiers;
    else
      nb_columns = 2;
  }

  std::vector<int64_t> dims{N, nb_columns};
  Tensor* Z = ctx->Output(1, TensorShape(dims));

  const auto x_data = X->template DataAsSpan<float>();
  int64_t zindex = 0;

  std::vector<float> scores_data;
  std::vector<float> kernels_data;
  std::vector<int64_t> votes_data;

  std::vector<float> classifier_scores_data;
  std::vector<float> probsp2_data;

  if (have_proba && mode_ == SVM_TYPE::SVM_SVC) {
    probsp2_data.resize(N * class_count_squared, 0.f);
  }

  // scores.reserve(class_count_squared);  // FIXME. N x class_count_ for linear. N x num_classifiers first for other path, then N x class_count_

  // need class_count_ per batch
  // votes.reserve(N * class_count_);

  // X: [num_batches, feature_count_] where features could be coefficients or support vectors
  // coefficients_: if linear [class_count, feature_count]
  //                else      [num_classes - 1, vector_count_]
  // support_vectors_ : [vector_count_, feature_count_]

  // Y: [num_batches, 1]
  // Z: [num_batches, class_count] ??? does write extra scores increase that to 2xclass_count?
  int64_t num_scores_per_batch = class_count_;

  if (mode_ == SVM_TYPE::SVM_LINEAR) {
    scores_data.resize(N * class_count_);

    // combine the coefficients with the input data and apply the kernel type
    auto out = gsl::make_span<float>(scores_data.data(), scores_data.size());
    batched_kernel_dot(x_data, coefficients_, N, class_count_, feature_count_, rho_[0], out, threadpool);

    //for (int64_t j = 0; j < class_count_; j++) {  //for each class
    //  auto val = kernel_dot(x_data, current_weight_0, coefficients_, feature_count_ * j,
    //                        feature_count_, get_kernel_type());
    //  val += rho_[0];
    //  scores.push_back(val);  // N x class_count_ (coefficients is {class_count,feature_count} so N,feature . feature, class
    //}
  } else {
    float* classifier_scores = nullptr;

    if (mode_ == SVM_TYPE::SVM_SVC && proba_.size() > 0) {
      // we will write N * num_classifiers scores first, and then reduce to N * class_count_, so need
      // to use a separate buffer for the first scoring.
      scores_data.resize(N * class_count_);
      classifier_scores_data.resize(N * num_classifiers);
      classifier_scores = classifier_scores_data.data();
    } else {
      // we will write directly to scores_data
      num_scores_per_batch = num_classifiers;
      scores_data.resize(N * num_classifiers);
      classifier_scores = scores_data.data();
    }

    kernels_data.resize(N * vector_count_);
    votes_data.resize(N * class_count_, 0);

    // combine the input data with the support vectors and apply the kernel type
    // output is {num_batches, vector_count_}
    auto kernels_span = gsl::make_span<float>(kernels_data.data(), kernels_data.size());
    batched_kernel_dot(x_data, support_vectors_, N, vector_count_, feature_count_, 0.f, kernels_span,
                       threadpool);

    for (int64_t n = 0; n < N; n++) {
      float* kernels = kernels_data.data() + (n * vector_count_);
      float* scores = classifier_scores + (n * num_classifiers);
      int64_t* votes = votes_data.data() + (n * class_count_);

      // reduce scores from kernels using coefficients, taking into account the varying number of support vectors
      // per class.
      // coefficients: [num_classes - 1, vector_count_]
      //
      // e.g. say you have 3 classes, with 3 x 3 coefficients
      //
      // AA AB AC
      // BA BB BC
      // CA CB CC
      //
      // you can remove the diagonal line of items comparing a class with itself leaving one less row.
      //
      // BA AB AC
      // CA CB BC
      //
      // for each class there is a coefficient per support vector, and a class has one or more support vectors.
      //
      // Combine the scores for the two combinations for two classes with their coefficient.
      // e.g. AB combines with BA.
      // If A has 3 support vectors and B has 2, there's a 3x2 block for AB and a 2x3 block for BA to combine
      //

      // votes.assign(class_count_, 0);
      for (int64_t i = 0; i < class_count_ - 1; i++) {
        int64_t start_index_i = starting_vector_[i];  // start of support vectors for class i
        int64_t class_i_support_count = vectors_per_class_[i];
        int64_t i_coeff_row_offset = vector_count_ * i;

        for (int64_t j = i + 1; j < class_count_; j++) {
          int64_t start_index_j = starting_vector_[j];  // start of support vectors for class j
          int64_t class_j_support_count = vectors_per_class_[j];
          int64_t j_coeff_row_offset = vector_count_ * (j - 1);

          double sum = 0;

          const float* val1 = &(coefficients_[j_coeff_row_offset + start_index_i]);
          const float* val2 = &(kernels[start_index_i]);
          for (int64_t m = 0; m < class_i_support_count; ++m, ++val1, ++val2)
            sum += *val1 * *val2;

          val1 = &(coefficients_[i_coeff_row_offset + start_index_j]);
          val2 = &(kernels[start_index_j]);

          for (int64_t m = 0; m < class_j_support_count; ++m, ++val1, ++val2)
            sum += *val1 * *val2;

          sum += rho_[i];  // rho_ entry for this classifier

          *scores = static_cast<float>(sum);
          ++scores;
          ++votes[sum > 0 ? i : j];
        }
      }

      assert(scores == classifier_scores + ((n + 1) * num_classifiers));
    }
  }

  for (int64_t n = 0; n < N; n++)  //for each example
  {
    float* _scores = scores_data.data() + (n * num_scores_per_batch);

    //!!!
    // temporary copy of scores into vector until other parts handle a gsl::span instead
    //!!!
    std::vector<float> scores(_scores, _scores + num_scores_per_batch);

    if (mode_ == SVM_TYPE::SVM_SVC && have_proba) {
      auto probsp2 = gsl::make_span<float>(probsp2_data.data() + (n * class_count_squared), class_count_squared);

      float* classifier_scores = classifier_scores_data.data() + (n * num_classifiers);

      int64_t index = 0;
      for (int64_t i = 0; i < class_count_ - 1; ++i) {
        int64_t p1 = i * class_count_ + i + 1;
        int64_t p2 = (i + 1) * class_count_ + i;
        for (int64_t j = i + 1; j < class_count_; ++j, ++index) {
          float val1 = sigmoid_probability(classifier_scores[index], proba_[index], probb_[index]);
          float val2 = std::max(val1, 1.0e-7f);
          val2 = std::min(val2, 1 - 1.0e-7f);
          probsp2[p1] = val2;
          probsp2[p2] = 1 - val2;
          ++p1;
          p2 += class_count_;
        }
      }

      multiclass_probability(class_count_, probsp2, scores);
    }

    float max_weight = 0;
    int64_t maxclass = -1;
    if (votes_data.size() > 0) {
      auto votes = gsl::make_span<int64_t>(votes_data.data() + (n * class_count_), class_count_);
      auto it_maxvotes = std::max_element(votes.cbegin(), votes.cend());
      maxclass = std::distance(votes.cbegin(), it_maxvotes);
    } else {
      auto it_max_weight = std::max_element(scores.cbegin(), scores.cend());
      maxclass = std::distance(scores.cbegin(), it_max_weight);
      max_weight = *it_max_weight;
    }

    // write top class
    // onnx specs expects one column per class.
    int write_additional_scores = -1;
    if (rho_.size() == 1) {
      if (using_strings_) {
        write_additional_scores = _set_score_svm<std::string>(Y, max_weight, maxclass, n, post_transform_, proba_,
                                                              weights_are_all_positive_, classlabels_strings_,
                                                              "1", "0");
      } else {
        write_additional_scores = _set_score_svm<int64_t>(Y, max_weight, maxclass, n, post_transform_, proba_,
                                                          weights_are_all_positive_, classlabels_ints_,
                                                          1, 0);
      }
    } else {  //multiclass
      if (using_strings_) {
        Y->template MutableData<std::string>()[n] = classlabels_strings_[maxclass];
      } else {
        Y->template MutableData<int64_t>()[n] = classlabels_ints_[maxclass];
      }
    }

    // we only write an extra score if there are 2 classes, and num_scores_per_batch == num_classifiers
    // which means there's only one entry in scores
    write_scores(scores, post_transform_, zindex, Z, write_additional_scores);
    zindex += scores.size();
  }

  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
