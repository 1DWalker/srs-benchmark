#include <vector>
#include <math.h>

// https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
struct adam {
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps = 1e-8f;
    const int n;
    float beta1_pow = 1.0f;
    float beta2_pow = 1.0f;
    std::vector<float>* params;
    std::vector<float> m, v;
    adam(std::vector<float>* _params): params(_params), n((int)_params->size()) {
        m.assign(n, 0.0f);
        v.assign(n, 0.0f);
    }
    void step(std::vector<float> &grad, float lr) {
        // weight decay step skipped
        beta1_pow *= beta1;
        beta2_pow *= beta2;
        for (int i = 0; i < n; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
            float m_hat = m[i] / (1 - beta1_pow);
            float v_hat = v[i] / (1 - beta2_pow);
            (*params)[i] -= lr * m_hat / (sqrt(v_hat) + eps);
        }
    }
};