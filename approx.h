#define APPROX_FN_CAST(fn) (void(*)(void*)) fn
#define APPROX_IO_CAST(fn) (void(*)(void*, double*)) fn

typedef struct {
  const char* name;
  const char* model_path;
  bool collect;
  bool infer;
  int input_dim;
  int output_dim;
  int (*get_N)(void *data);
  void (*fill_input)(void* data, double *input);
  void (*fill_output)(void* data, double *output);
  void (*apply_output)(void* data, double *output);
} ApproxConfig;

void wrap_approx(void (*func)(void *data), void *data, ApproxConfig config);

extern ApproxConfig Config_CalcForceForNodes;
extern ApproxConfig Config_LagrangeNodal;
