#include <torch/extension.h>

template<typename T>
struct StateBuffer {
    torch::Tensor storage_u8;
    int64_t capacity = 0; // number of fsrs_state_t slots

    T* ensure(int64_t req_size, const torch::Tensor& like) {
        if (req_size <= capacity) {
            return reinterpret_cast<fsrs_state_t*>(storage_u8.data_ptr<uint8_t>());
        }

        int64_t new_capacity = std::max<int64_t>(
            req_size,
            static_cast<int64_t>(std::ceil(req_size * 1.1))
        );

        int64_t nbytes = new_capacity * static_cast<int64_t>(sizeof(T));

        storage_u8 = torch::empty(
            {nbytes},
            like.options().dtype(torch::kUInt8)
        );

        capacity = new_capacity;

        return reinterpret_cast<fsrs_state_t*>(storage_u8.data_ptr<uint8_t>());
    }
};