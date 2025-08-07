// Shorthands to index into the global gradient array of shape [M, M, 6]
// I experimented with all memory layouts and this one results in the most
// coalesced writes
//

// nvrtc is missing cstdint
typedef signed char int8_t;
typedef long long int64_t;

#define P 7

#define LOG_B 0
#define LOG_D 1
#define LOG_U 2
#define LOG_V 3
#define LOG_E0 4
#define LOG_E1 5
#define LOG_PI 6

#define LOG_X(m, i)   H[i * M * P + m * P + g]  // [M,M,7]
#define LOG_X_STRIDE  M * P

// Shorthands to index into the global params array
#define B(m) p[0 * M + m]
#define D(m) p[1 * M + m]
#define U(m) p[2 * M + m]
#define V(m) p[3 * M + m]
#define EMIS(ob, m) p[(4 + ob) * M + m]
#define PI(m) p[6 * M + m]

template<typename T>
__device__ void matvec(T *v, FLOAT *p, int stride = 1) {
    // in-place O(M) matrix-vector multiply with transition matrix
    FLOAT tmp[M];
    FLOAT x, sum;
    int i;
    sum = 0.;
    for (i = 0; i < M; ++i) {
        x = *(v + stride * i);
        tmp[i] = x * D(i) + sum * V(i);
        sum += U(i) * x;
    }
    sum = 0.;
    for (i = M - 1; i >= 0; --i) {
        x = *(v + stride * i);
        *(v + stride * i) = tmp[i] + sum * B(i);
        sum += x;
    }
}

__device__ FLOAT p_emis(const int8_t ob, FLOAT *p, const int m) {
    if (ob == -1) return 1.;
    return EMIS(ob, m);
}

/*

extern "C"
__global__ void
// log-likelihood function without gradient
loglik(int8_t const *datag,
       const int64_t L,
       FLOAT const *pa,  // [B, P, M]
       double *loglik
       ) {
    const int64_t b = blockIdx.x;
    const int64_t m = threadIdx.x;

    __shared__ FLOAT p[P * M];
    FLOAT h[M], c;
    int m;

    using WarpReduce = cub::WarpReduce<FLOAT>;
    // Allocate WarpReduce shared memory for 1 warp
    __shared__ typename WarpReduce::TempStorage temp_storage[1];

    // copy local global parameters to local
    FLOAT const *pab = &pa[b * P * M];
    if (s == 0) {
        memcpy(p, pab, P * M * sizeof(FLOAT));
    }
    __syncthreads();
    h[m] = PI(m);  // initialize to pi
    __syncthreads();
    double ll = 0.;
    // local variables
    const int8_t *data = &datag[b * L];
    int8_t ob;
    int64_t ell;
    for (ell = 0; ell < L;  ell++) {
        if (m == 0)
            matvec(h, p);
        __syncthreads();
        ob = data[ell];
        h[m] *= p_emis(ob, p, m);
        __syncthreads();
        c = WarpReduce(temp_storage[0]).Sum(h[m]);
        __syncthreads();
        h[m] /= c[0];
        __syncthreads();
        ll += log(c);
    }
    loglik[b] = ll;
}

*/

extern "C"
__global__ void
__launch_bounds__(7 * M)
// value and gradient of the log-likelihood function
loglik_grad(int8_t const *datag,
          const int64_t L,
          FLOAT const *pa,
          FLOAT *loglik,
          FLOAT *dlog
         ) {
    const int64_t b = blockIdx.x;
    const int64_t g = threadIdx.x;
    const int64_t m = threadIdx.y;

    __shared__ FLOAT H[P * M * M];
    __shared__ int8_t data_buf[P * M];  // buffer for data
    __shared__ FLOAT p[P * M];
    __shared__ FLOAT h[M];
    __shared__ FLOAT c;
    FLOAT ll;

    // copy local global parameters to local
    FLOAT const *pab = &pa[b * P * M];
    if (g == 0) {
        if (m == 0) {
            memcpy(p, pab, P * M * sizeof(FLOAT));
            memset(H, 0., sizeof(FLOAT) * P * M * M);
            memset(dlog, 0., sizeof(FLOAT) * P * M * M);
            c = 0.;
            ll = 0.;
        }
        h[m] = PI(m);
    }
    __syncthreads();
    // initialize the H matrix for pi to identity; all others zero.
    // (and technically, it's the gradient w/r/t pi not log_pi)
    LOG_X(m, m) = FLOAT(g == LOG_PI);
    __syncthreads();
    if (g == 0) h[m] = PI(m);  // initialize to pi
    __syncthreads();
    // local variables
    int8_t ob;
    int i, j;
    FLOAT tmp;
    ll = 0.;
    FLOAT sum1, sum2;
    int start, delta;
    if (g == LOG_B) {
        start = M - 1;
        delta = -1;
    } else {
        start = 0;
        delta = +1;
    }
    // main data loop
    const int8_t *data = &datag[b * L];
    int64_t ell, u, v, w;
    c = 0.;
    ell = 0;
    __syncthreads();
    for (u = 0; u < (L + P * M - 1) / (P * M); u++) {
        // read a chunk of data into shared memory
        w = (u * P * M) + g * M + m;
        if (w < L)
            data_buf[g * M + m] = data[w];
        __syncthreads();
        for (v = 0; v < P * M; v++) {
            if (ell++ >= L) break;
            ob = data_buf[v];
            // update each derivative matrix
            matvec(&LOG_X(m, 0), p, LOG_X_STRIDE);
            sum1 = 0.;
            sum2 = 0.;
            for (j = start; j >= 0 && j < M; j += delta) {
                // B counts down and everything else counts up
                tmp = (
                      // diag(hr * b)
                      (g == LOG_B) * (m == j) * (sum1 * B(j)) +
                      // diag(h * d)
                      (g == LOG_D) * (m == j) * (D(j) * h[j]) +
                      //
                      (g == LOG_U) * (m < j) * (h[m] * U(m) * V(j)) +
                      //
                      (g == LOG_V) * (j - 1 == m) * (sum2 * V(j))
                );
                sum1 += h[j];
                sum2 += U(j) * h[j];
                LOG_X(m, j) += tmp;
            }
            __syncthreads();
            if (g == 0 && m == 0) matvec(h, p);
            __syncthreads();
            LOG_X(m, m) += (g == LOG_E0) * h[m] * (ob == 0);
            LOG_X(m, m) += (g == LOG_E1) * h[m] * (ob == 1);
            __syncthreads();
            if (g == 0) {
                h[m] *= p_emis(ob, p, m);
                atomicAdd(&c, h[m]);
            }
            __syncthreads();
            if (g == 0) {
                h[m] /= c;
            }
            for (j = 0; j < M; ++j) {
                LOG_X(m, j) *= p_emis(ob, p, j) / c;
            }
            __syncthreads();
            if (g == 0 && m == 0) {
                ll += log(c);
                c = 0.;
            }
            __syncthreads();
        }
    }
    __syncthreads();
    if (g == 0 && m == 0) {
        loglik[b] = ll;
    }
    // for pi we accumulated dll/dpi instead of dll/dlog(pi) so
    // we have to multiply by
    const FLOAT x = (g == LOG_PI) ? PI(m) : 1.;
    for (i = 0; i < M; ++i) {
        dlog[
            b * P * M + 
            g * M + 
            + m
        ] += LOG_X(m, i) * x;
    }
}
