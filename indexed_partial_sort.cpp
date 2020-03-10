#include <iostream>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort
#include <thread>

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



template <typename T>
vector<size_t> partial_sort_indexes_distributed(const vector<T> &v, int k, int nthreads) {
    // initialize original index locations
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);


    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values

    //unsigned int nthreads = std::thread::hardware_concurrency();
    vector<size_t> smaller_idx(nthreads*k);
    size_t batch_size = v.size() / nthreads;
    // TODO assert();
    
    #pragma omp parallel for
    for (int i=0;i<nthreads;i++){
        auto start = idx.begin() + i*batch_size;
        auto end = std::max(start + batch_size, idx.end());
        partial_sort(start, start + k, end,
                     [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
        smaller_idx.insert(smaller_idx.begin() + i*k , start, start + k);
    }

    partial_sort(smaller_idx.begin(), smaller_idx.begin() + k, smaller_idx.end(),
                 [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    smaller_idx.resize(k);
    return smaller_idx;
}





template <typename T>
vector<size_t> n_element_and_sort(const vector<T> &v, int k) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  nth_element(idx.begin(), idx.begin()+k, idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
  idx.resize(k);
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
  return idx;
}


//// Driver program to test above methods 
//int main()
//{
//        int arr[] = { 10, 4, 5, 0 , 98,121,12,0,12,-2,-5, 8, 6, 26, 11 };
//        int n = sizeof(arr) / sizeof(arr[0]);
//        vector<int> vect(arr, arr + n);
//        for (int i=0;i<n;i++)
//            cout << arr[i] << " ";
//        cout << endl;
//        int k = 3;
//        vector<size_t> temp = partial_sort_indexes_distributed(vect,k);
//        for (size_t i=0;i<k;i++)
//            cout << arr[temp.at(i)] << " ";
//        cout << endl;
//        return 0;
//}
//
