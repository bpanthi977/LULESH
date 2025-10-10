#define APPROX_FN_CAST(fn) (void(*)(void*)) fn
#define APPROX_IO_CAST(fn) (void(*)(void*, double*)) fn

typedef struct {
  const char* name;
  const char* model_path;
  const char* db_path;
  const int collect_every;
  bool collect;
  bool infer;
  int input_dim;
  int output_dim;
  int (*get_N)(void *data);
  void (*fill_input)(void* data, double *input);
  void (*fill_output)(void* data, double *output);
  void (*apply_output)(void* data, double *output);
  int funcall_counter;
} ApproxConfig;

void wrap_approx(void (*func)(void *data), void *data, ApproxConfig *config);

extern ApproxConfig Config_CalcForceForNodes;
extern ApproxConfig Config_LagrangeNodal;
