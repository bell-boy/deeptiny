#pragma once

namespace deeptiny {

struct State {
  bool grad_enabled = true;
};

extern State GradState;

class GradMode {
 public:
  explicit GradMode(bool enabled) : prev_(GradState.grad_enabled) {
    GradState.grad_enabled = enabled;
  }
  ~GradMode() { GradState.grad_enabled = prev_; }

  GradMode(const GradMode&) = delete;
  GradMode& operator=(const GradMode&) = delete;

 private:
  bool prev_;
};

class NoGrad {
 public:
  NoGrad() : guard_(false) {}

 private:
  GradMode guard_;
};

}  // namespace deeptiny
