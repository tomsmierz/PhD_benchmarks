
extern "C"{ // wymagane , ponieważ c++ naturalnie dekoruje (name mangling) nazwy funkcji, przez co trudno je potem linkować


__global__ void parallel_annealing_step(float* A, float* h, float* x, float* momentum, // macierze wejściowe
                                        float lambda_t, float step_size, int size, // stałe
                                        float* momentum_new, float* x_new, float* state_new) { // macierze wyjsciowe

    float beta = 1.0F - step_size;

    int ti = threadIdx.x;  // pojedyńczy element w kolumnie
    int col = blockIdx.x;  // każdy blok zajmuje się jedną kolumną (trajektorią)
    int k = blockIdx.y;  // ewentualne dodatkowe bloki na kolumne
    int blockSize = blockDim.x; // rozmiar bloku

    int N = size;

    // gdzie "globalnie" jesteśmy w macierzy
    int global_row = ti + k * blockSize;
    int num_cols = gridDim.x;
    

    if (global_row < N){
        float x_ij = x[global_row * num_cols + col];

        float gradient = -1.0F * A[global_row * num_cols + col] - h[global_row] + x_ij * lambda_t;
        float momentum_value = momentum[global_row * num_cols + col] * beta - step_size * gradient;  // momentum 

        float momentum_clipped = max(-1.0F, min(1.0F, momentum_value));
        float x_new_clipped =  max(-1.0F, min(1.0F, x_ij + momentum_clipped));

        momentum_new[global_row * num_cols + col] = momentum_clipped;  // nowe momentum
        x_new[global_row * num_cols + col] = x_new_clipped;  // nowy x
        // Rzutowanie stanu
        if (x_new_clipped > 0){
            state_new[global_row * num_cols + col] = 1.0F;
        } else if (x_new_clipped < 0){
            state_new[global_row * num_cols + col] = -1.0F;
        } else {
            state_new[global_row * num_cols + col] = 0.0F;
        }

    }    
}



__global__ void parallel_annealing_step_rowmajor(
                                    const float* __restrict__ A,     // (n, M)
                                    const float* __restrict__ h,     // (n,)
                                    float* __restrict__ x,           // (n, M)
                                    float* __restrict__ m,           // (n, M)
                                    float lambda_t,
                                    float step_size,
                                    int n,
                                    int M,
                                    float* __restrict__ state        // (n, M)
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= M) return;

    int idx = row * M + col;

    float xij = x[idx];
    float mij = m[idx];

    // gradient = -A - h + lambda*x
    float grad = -A[idx] - h[row] + xij * lambda_t;

    float beta = 1.0f - step_size;

    // m_new = beta*m - step_size*grad  (use fma)
    float mnew = fmaf(beta, mij, -step_size * grad);

    // clip momentum to [-1, 1]
    mnew = fminf(1.0f, fmaxf(-1.0f, mnew));

    // x_new = clip(x + m_new, [-1,1])
    float xnew = xij + mnew;
    xnew = fminf(1.0f, fmaxf(-1.0f, xnew));

    // write back
    m[idx] = mnew;
    x[idx] = xnew;

    // branchless sign projection to {-1, 0, +1}
    // (bools become 0/1)
    state[idx] = (xnew > 0.0f) - (xnew < 0.0f);
}
}


// __global__ void parrarel_annealing_step(float A[N][M], float h[N], float x[N][M], float momentum[N][M], // macierze wejściowe
//                                         float lambda_t, float step_size,  // stałe
//                                         float momentum_new[N][M], float x_new[N][M], float state_new[N][M]) { // macierze wyjsciowe

//     float beta = 1.0F - step_size;

//     int ti = threadIdx.x;  // pojedyńczy element w kolumnie
//     int col = blockIdx.x;  // każdy blok zajmuje się jedną kolumną (trajektorią)
//     int k = blockIdx.y;  // ewentualne dodatkowe bloki na kolumne
//     int blockSize = blockDim.x; // rozmiar bloku

//     // gdzie "globalnie" jesteśmy w macierzy
//     int global_row = ti + k * blockSize;

//     if (global_row < N){
//         float x_ij = x[global_row][col];

//         float gradient = -1.0F * A[global_row][col] - h[global_row] + x_ij * lambda_t;
//         float momentum_value = momentum[global_row][col] * beta - step_size * gradient;  // momentum 

//         float momentum_clipped = max(-1.0F, min(1.0F, momentum_value));
//         float x_new_clipped =  max(-1.0F, min(1.0F, x_ij + momentum_clipped));

//         momentum_new[global_row][col] = momentum_clipped;  // nowe momentum
//         x_new[global_row][col] = x_new_clipped;  // nowy x
//         // Rzutowanie stanu
//         if (x_new_clipped > 0){
//             state_new[global_row][col] = 1.0F;
//         } else if (x_new_clipped < 0){
//             state_new[global_row][col] = -1.0F;
//         } else {
//             state_new[global_row][col] = 0.0F;
//         }

//     }    
// }




