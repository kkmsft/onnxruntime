// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

// Rewrite rule that eliminates the identity node.
class EliminateIdentity : public RewriteRule {
 public:
  EliminateIdentity() noexcept : RewriteRule("EliminateIdentity", "Eliminate identity node") {}

 private:
  bool SatisfyCondition(const Node& node) override;

  Status Apply(Graph& graph_editor, Node& node, bool& modified) override;
};

}  // namespace onnxruntime
