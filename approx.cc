#include "lulesh.h"
#include "approx.h"

/// ANNOTATE

void wrap_approx(void (*func)(void *data), void *data, ApproxConfig *config) {
  double *input;
  double *output;
  bool do_collect = config->collect && config->funcall_counter % config->collect_every == 0;
  if (do_collect || config->infer) {
    int N = config->get_N(data);
    input = (double*) malloc(sizeof(double) * N * config->input_dim);
    output = (double*) malloc(sizeof(double) * N * config->output_dim);
    config->fill_input(data, input);
  }

  if (do_collect) {
    printf("Collecting %d\n", config->funcall_counter);
    int N = config->get_N(data);
    int n_input = config->input_dim;
    int n_output = config->output_dim;

#pragma approx declare tensor_functor(identity_map_2d: [i,j] = ([i,j]))
#pragma approx declare tensor(inp: identity_map_2d(input[0:N, 0:n_input]))
#pragma approx ml(offline) in(inp) out(identity_map_2d(output[0:N, 0:n_output])) label(config->name)
    {
      func(data);
      config->fill_output(data, output);
    }
    free(output);
    free(input);
  } else if (config->infer) {
    int N = config->get_N(data);
    int n_input = config->input_dim;
    int n_output = config->output_dim;

#pragma approx declare tensor_functor(identity_map_2d: [i,j] = ([i,j]))
#pragma approx declare tensor(inp: identity_map_2d(input[0:N, 0:n_input]))
#pragma approx ml(infer) in(inp) out(identity_map_2d(output[0:N, 0:n_output])) label(config->name) model(config->model_path)
    {
      func(data);
      config->apply_output(data, output);
    }
    free(output);
    free(input);
  } else {
    func(data);
  }

  config->funcall_counter += 1;
}

// Helpers

typedef struct {
  double x;
  double y;
  double z;
  double xd;
  double yd;
  double zd;
  double nodalMass;
} ln_node_input;

typedef struct {
  double q;
  double volo;
  double v;
  double elemMass;
  double ss;
} ln_elem_input;

typedef struct {
  // Main node is connected (belongs to) to 8 elements
  ln_node_input main_node;
  ln_node_input elems_other_nodes[8][7];  // 7 nodes of the 8 elements
  ln_elem_input elems[8]; // 8 elements
} ln_input;

void setNodeValues(Domain& domain, int i, ln_node_input* input) {
  input->x = domain.x(i);
  input->y = domain.y(i);
  input->z = domain.z(i);
  input->xd = domain.xd(i);
  input->yd = domain.yd(i);
  input->zd = domain.zd(i);
  input->nodalMass = domain.nodalMass(i);
}

void setElementValues(Domain& domain, int i, ln_elem_input* input) {
  input->q = domain.q(i);
  input->volo = domain.volo(i);
  input->v = domain.v(i);
  input->elemMass = domain.elemMass(i);
  input->ss = domain.ss(i);
}

void setNodeNeighbours(Domain& domain, int node_i, Index_t* node_idxs, ln_node_input* input) {
  int count = 0;
  for (int k=0; k<8; k++) {
    Index_t node_idx = node_idxs[k];
    if (node_idx == node_i) {
      continue;
    }
    setNodeValues(domain, node_i, &input[count]);
    count++;
  }
}

void fill_output(double *output, std::vector<Real_t> *vectors, int N, int n_vector) {
  for (int i=0; i<N; i++) {
    for (int j=0; j<n_vector; j++) {
      *(output++) = vectors[j][i];
    }
  }
}

void apply_output(double *output, std::vector<Real_t> *vectors, int N, int n_vector) {
  for (int i=0; i<N; i++) {
    for (int j=0; j<n_vector; j++) {
      vectors[j][i] = *(output++);
    }
  }
}

int get_nodes_n(Domain &domain) { return domain.m_numNode; }

/// FORCE

void fill_input_node(Domain& domain, double* _input) {
  ln_input *input = (ln_input*) _input;

  int *elemFillCount = (int*)calloc(domain.m_numNode, sizeof(int));
#pragma omp parallel for
  for (int i=0; i<domain.m_numNode; i++) {
    setNodeValues(domain, i, &input[i].main_node);
  }

  for (int elem_i=0; elem_i<domain.m_numElem; elem_i++) { // elements
    for (int j=0; j<8; j++) { // nodes of the element i
      int node_j = domain.nodelist(elem_i)[j];
      int elem_offset = elemFillCount[node_j];
      setNodeNeighbours(domain, node_j, domain.nodelist(elem_i), input[node_j].elems_other_nodes[elem_offset]);
      setElementValues(domain, elem_i, &input[node_j].elems[elem_offset]);
      elemFillCount[node_j]++;
    }
  }

  free(elemFillCount);
}

void fill_output_force(Domain &domain, double *output) {
  std::vector<Real_t> vectors[3] = {domain.m_fx, domain.m_fy, domain.m_fz};
  fill_output(output, vectors, domain.m_numNode, 3);
}

void apply_output_force(Domain& domain, double *output) {
  std::vector<Real_t> vectors[3] = {domain.m_fx, domain.m_fy, domain.m_fz};
  fill_output(output, vectors, domain.m_numNode, 3);
}

ApproxConfig Config_CalcForceForNodes = {
  .name = "CalcForceForNodes",
  .model_path = "calc_force.pt",
  .db_path = "Force.h5",
  .collect_every = 100,
  #ifdef COLLECT_FORCE
  .collect = true,
  #else
  .collect = false,
  #endif

  #ifdef INFER_FORCE
  .infer = true,
  #else
  .infer = false,
  #endif


  .input_dim = sizeof(ln_input) / sizeof(double),
  .output_dim = 3,
  .get_N = (int(*)(void*))get_nodes_n,

  .fill_input = APPROX_IO_CAST(fill_input_node),
  .fill_output =  APPROX_IO_CAST(fill_output_force),
  .apply_output = APPROX_IO_CAST(apply_output_force),
  .funcall_counter = 0
};
