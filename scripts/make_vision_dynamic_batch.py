# -*- coding: utf-8 -*-
"""Repair baked-in batch=1 reshape constants in the split TurboVLA vision ONNX.

The traced export hardcoded ``batch * cameras`` (3 with one sample) as a
constant leading dimension in the attention reshapes, so TensorRT silently
builds a static batch=1 engine even when given a 1..4 optimization profile.
This script rewrites those reshape targets to ``Concat([Shape(images)[0] * 3],
tail)`` so the batch dimension stays dynamic end to end.

Usage:
    python scripts/make_vision_dynamic_batch.py INPUT.onnx OUTPUT.onnx [--cameras 3]
"""
from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _const_map(graph):
    values = {init.name: init for init in graph.initializer}
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                values[node.output[0]] = numpy_helper.to_array(attr.t)
    return values


def _resolve(values, name):
    value = values.get(name)
    if value is None:
        return None
    if isinstance(value, TensorProto):
        if value.data_location == TensorProto.EXTERNAL:
            return None
        return numpy_helper.to_array(value)
    return value


def repair_dynamic_batch(input_path: str, output_path: str, input_name: str,
                         cameras: int) -> int:
    model = onnx.load(input_path)
    graph = model.graph

    shape_out = None
    shape_index = 0
    for index, node in enumerate(graph.node):
        if node.op_type == "Shape" and node.input and node.input[0] == input_name:
            shape_out = node.output[0]
            shape_index = index
            break
    if shape_out is None:
        shape_out = "batch_repair/shape_out"
        graph.node.insert(0, helper.make_node(
            "Shape", [input_name], [shape_out], name="batch_repair/Shape"))
        shape_index = 0

    def add_init(name, array):
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(array, dtype=np.int64), name))
        return name

    idx0 = add_init("batch_repair/idx0", [0])
    cams = add_init("batch_repair/cameras", [cameras])
    graph.node.insert(shape_index + 1, helper.make_node(
        "Gather", [shape_out, idx0], ["batch_repair/b"], axis=0,
        name="batch_repair/GatherBatch"))
    graph.node.insert(shape_index + 2, helper.make_node(
        "Mul", ["batch_repair/b", cams], ["batch_repair/bc"],
        name="batch_repair/MulCameras"))

    values = _const_map(graph)
    fixed = 0
    for index, node in enumerate(list(graph.node)):
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape = _resolve(values, node.input[1])
        if shape is None:
            continue
        shape = np.asarray(shape).flatten()
        if len(shape) < 2 or int(shape[0]) != cameras:
            continue
        tail = add_init(f"batch_repair/tail_{fixed}", shape[1:].tolist())
        dyn = f"batch_repair/shape_{fixed}"
        graph.node.insert(index + 1 + fixed, helper.make_node(
            "Concat", ["batch_repair/bc", tail], [dyn], axis=0,
            name=f"batch_repair/Concat_{fixed}"))
        node.input[1] = dyn
        fixed += 1
    if fixed == 0:
        raise RuntimeError(f"no baked batch reshapes found in {input_path}")
    onnx.save(model, output_path)
    return fixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--cameras", type=int, default=3)
    args = parser.parse_args()
    fixed = repair_dynamic_batch(args.input, args.output, args.input_name,
                                 args.cameras)
    print(f"repaired {fixed} reshape constants -> {args.output}")


if __name__ == "__main__":
    main()