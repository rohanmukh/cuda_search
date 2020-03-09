#include <iostream>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

using namespace std;

template <typename T>
vector<size_t> partial_sort_indexes(const vector<T> &v, int k) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  partial_sort(idx.begin(), idx.begin()+k, idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

  idx.resize(k);
  return idx;
}
