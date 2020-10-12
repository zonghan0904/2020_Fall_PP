#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x;
  __pp_vec_int y;
  __pp_vec_float result;
  __pp_vec_int count = _pp_vset_int(0);
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_float max = _pp_vset_float(9.999999f);
  __pp_vec_float comp_x = _pp_vset_float(0.f);
  __pp_vec_int comp_y = _pp_vset_int(1);
  __pp_mask maskAll, maskIsEqual, maskIsNotEqual;
  __pp_mask maskIsPositive;
  __pp_mask maskIsLarge;
  __pp_mask maskNEAndP;
  __pp_mask maskNEAndL;

  int mod = N % VECTOR_WIDTH;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsEqual = _pp_init_ones(0);

    // To handle (N % VECTOR_WIDTH) != 0
    if ((i + VECTOR_WIDTH) > N){
	for (int j = 0; j < mod; j++){
	    comp_x.value[j] = values[i+j];
	    comp_y.value[j] = exponents[i+j];
	}

	// Load vector of values from contiguous memory addresses
    	_pp_vmove_float(x, comp_x, maskAll); // x = values[i];

    	// Load vector of exponents from contiguous memory addresses
    	_pp_vmove_int(y, comp_y, maskAll); // x = values[i];
    }
    else{
	// Load vector of values from contiguous memory addresses
    	_pp_vload_float(x, values + i, maskAll); // x = values[i];

    	// Load vector of exponents from contiguous memory addresses
    	_pp_vload_int(y, exponents + i, maskAll); // x = values[i];
    }

    // Set mask according to predicate
    _pp_veq_int(maskIsEqual, y, zero, maskAll); // if (y == 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vset_float(result, 1.f, maskIsEqual);   //   output[i] = 1.f;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotEqual = _pp_mask_not(maskIsEqual); // } else {

    // Execute instruction ("else" clause)
    _pp_vmove_float(result, x, maskIsNotEqual); //     result = x;
    _pp_vsub_int(count, y, one, maskIsNotEqual);//     count = y - 1;}
    _pp_vgt_int(maskIsPositive, count, zero, maskIsNotEqual);

    // Loop instruction("While" loop)
    while (_pp_cntbits(maskIsPositive)){
	maskNEAndP = _pp_mask_and(maskIsNotEqual, maskIsPositive);
	_pp_vmult_float(result, result, x, maskNEAndP);	// result *= x;
	_pp_vsub_int(count, count, one, maskNEAndP);	// count--;
	_pp_vgt_int(maskIsPositive, count, zero, maskNEAndP);
    }

    // Set mask according to predicate
    _pp_vgt_float(maskIsLarge, result, max, maskIsNotEqual); // if (result > 9.999999f) {

    // Execute instruction using mask ("if" clause)
    maskNEAndL = _pp_mask_and(maskIsNotEqual, maskIsLarge);
    _pp_vset_float(result, 9.999999f, maskNEAndL);   //   output[i] = 9.999999f;

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);

  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  float sum = 0.f;
  __pp_vec_float x, hadd_result, inter_result;
  __pp_mask maskAll;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
      // All ones
      maskAll = _pp_init_ones();

      _pp_vload_float(x, values + i, maskAll);
      _pp_hadd_float(hadd_result, x);
      _pp_interleave_float(inter_result, hadd_result);

      for (int j = 0; j < VECTOR_WIDTH / 2; j++){
	  sum += inter_result.value[j];
      }
  }

  return sum;
}
