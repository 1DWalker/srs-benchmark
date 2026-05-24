#include <cuda_runtime.h>

struct fsrs_state_t {
    float s; 
    float d;
};

struct fsrs_params_t {

};

fsrs_state_t fsrs7_init(
    const fsrs_params_t &fsrs_params,
    const int8_t first_rating
) {

}

__device__
fsrs_state_t fsrs7_step(
    const fsrs_params_t &fsrs_params,
    const fsrs_state_t fsrs_state,
    const T elapsed_time,
    const int8_t rating
) {

}