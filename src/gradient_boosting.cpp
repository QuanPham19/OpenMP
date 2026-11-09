// Capstone: Gradient Boosting Histogram

// PROBLEM STATEMENT
// For every feature 'f' and bin 'b', sum all grad/hess in a node if in bin 'b'

// INPUT:
// - bins[node][feature]
// - grad[node]
// - hess[node]

// OUTPUT:
// - flat global histogram array
// - attribute: hist.grad, hist.hess, hist.cut, hist.offset