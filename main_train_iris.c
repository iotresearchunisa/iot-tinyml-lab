#include <stdio.h>
#include "aifes.h"
#include "iris_dataset.h"

void print_tensor_like_array(const char *name, const aitensor_t *tensor) {
    uint32_t total = 1;
    for (uint32_t i = 0; i < tensor->dim; i++) {
        total *= tensor->shape[i];
    }

    printf("float %s[%u] = {", name, total);
    for (uint32_t i = 0; i < total; i++) {
        printf("%ff", ((float *) tensor->data)[i]);
        if (i < total - 1)
            printf(", ");
    }
    printf("};\n");
}


int main(int argc, char *argv[]) {
    // Dataset Preparation
    uint16_t training_input_shape[] = {120, 4};
    uint16_t training_target_shape[] = {120, 3};

    aitensor_t training_input_tensor = AITENSOR_2D_F32(training_input_shape, training_features);
    aitensor_t training_target_tensor = AITENSOR_2D_F32(training_target_shape, training_target);

    // Neural Network Design
    aimodel_t model;

    uint16_t input_layer_shape[] = {1, 4};
    ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(4, input_layer_shape);

    ailayer_dense_f32_t dense_layer_1 = AILAYER_DENSE_F32_A(32);
    ailayer_relu_f32_t relu_layer_1 = AILAYER_RELU_F32_A();

    ailayer_dense_f32_t dense_layer_2 = AILAYER_DENSE_F32_A(16);
    ailayer_relu_f32_t relu_layer_2 = AILAYER_RELU_F32_A();

    ailayer_dense_f32_t dense_layer_3 = AILAYER_DENSE_F32_A(3);
    ailayer_softmax_f32_t softmax_layer_3 = AILAYER_SOFTMAX_F32_A();

    ailayer_t *x;

    model.input_layer = ailayer_input_f32_default(&input_layer);
    x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
    x = ailayer_relu_f32_default(&relu_layer_1, x);
    x = ailayer_dense_f32_default(&dense_layer_2, x);
    x = ailayer_relu_f32_default(&relu_layer_2, x);
    x = ailayer_dense_f32_default(&dense_layer_3, x);
    x = ailayer_softmax_f32_default(&softmax_layer_3, x);
    model.output_layer = x;

    // Neural Network Compilation
    ailoss_crossentropy_t loss_fun;
    model.loss = ailoss_crossentropy_f32_default(&loss_fun, model.output_layer);

    aialgo_compile_model(&model);

    uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
    void *parameter_memory = malloc(parameter_memory_size);
    if (parameter_memory == NULL) {
        return 0;
    }

    aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);

    aimath_f32_default_init_glorot_uniform(&dense_layer_1.weights);
    aimath_f32_default_init_zeros(&dense_layer_1.bias);

    aimath_f32_default_init_glorot_uniform(&dense_layer_2.weights);
    aimath_f32_default_init_zeros(&dense_layer_2.bias);

    aimath_f32_default_init_glorot_uniform(&dense_layer_3.weights);
    aimath_f32_default_init_zeros(&dense_layer_3.bias);

    aiopti_adam_f32_t adam_opti = AIOPTI_ADAM_F32(0.1f, 0.9f, 0.999f, 1e-7);
    aiopti_t *optimizer = aiopti_adam_f32_default(&adam_opti);

    uint32_t memory_size = aialgo_sizeof_training_memory(&model, optimizer);
    void *memory_ptr = malloc(memory_size);
    if (memory_ptr == NULL) {
        return 0;
    }

    aialgo_schedule_training_memory(&model, optimizer, memory_ptr, memory_size);

    aialgo_init_model_for_training(&model, optimizer);

    //Training
    uint16_t epochs = 25;
    uint32_t batch_size = 12;
    float lossa;
    printf("Start training\n");
    for (uint32_t i = 0; i < epochs; i++) {
        aialgo_train_model(&model, &training_input_tensor, &training_target_tensor, optimizer, batch_size);
        aialgo_calc_loss_model_f32(&model, &training_input_tensor, &training_target_tensor, &lossa);
        printf("Epoch %5d: loss: %f\n", i, lossa);
    }
    printf("Finished training\n");

    //Testing
    uint16_t input_shape_testing[] = {1, 4};

    for (int i = 0; i<30; i++) {
        aitensor_t input_tensor_testing = AITENSOR_2D_F32(input_shape_testing, (testing_features + i * 4));
        aitensor_t *out = aialgo_forward_model(&model, &input_tensor_testing);

        for (uint32_t y = 0; y < 3; y++) {
            printf("%f, %f\n", *(testing_target + (i * 3) + y), ((float*)out->data)[y]);
        }
        printf("\n");
    }

    // Neural Network Export
    print_tensor_like_array("weights_data_dense_1", &(dense_layer_1.weights));
    print_tensor_like_array("bias_data_dense_1", &(dense_layer_1.bias));
    print_tensor_like_array("weights_data_dense_2", &(dense_layer_2.weights));
    print_tensor_like_array("bias_data_dense_2", &(dense_layer_2.bias));
    print_tensor_like_array("weights_data_dense_3", &(dense_layer_3.weights));
    print_tensor_like_array("bias_data_dense_3", &(dense_layer_3.bias));

    //Memory free
    free(parameter_memory);
    free(memory_ptr);
}
