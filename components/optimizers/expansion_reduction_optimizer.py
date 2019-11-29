from components.transformation_graph import *
from components.optimizers.base_optimizer import Optimizer
from components.transformers.generator.polynomial_generator import PolynomialTransformation
from components.transformers.selector.variance_selector import VarianceSelector
from components.transformers.generator.pca_decomposer import PcaDecomposer
from components.transformers.merger import Merger


class ExpansionReductionOptimizer(Optimizer):
    def __init__(self, input_data: DataNode):
        super().__init__(str(__class__), input_data)

    def optimize(self):
        trans_set = self.get_available_transformations(self.root_node, trans_types=[1, 2, 3, 4])
        nodes = [self.root_node]
        input_node = self.root_node

        for transformer in trans_set:
            output_node = transformer.operate(input_node)
            nodes.append(output_node)

            self.graph.add_node(output_node)
            self.graph.add_trans_in_graph(input_node, output_node, transformer)

        # 1. Apply polynomial transformation on the non-categorical features.
        # 2. Conduct feature selection.
        if input_node.shape[1] - input_node.cat_num > 1:
            input_node = self.root_node
            transformer = PolynomialTransformation()
            try:
                output_node = transformer.operate(input_node)
                nodes.append(output_node)
                print('Shape ==>', input_node.shape, output_node.shape)
                self.graph.add_node(output_node)
                self.graph.add_trans_in_graph(input_node, output_node, transformer)
            except MemoryError as e:
                print(e)

        # 1. Apply cross transformations on the categorical features.
        # 2. Conduct feature selection.
        if input_node.cat_num > 1:
            input_node = self.root_node
            transformer = PolynomialTransformation()
            transformer.input_type = CATEGORICAL
            transformer.output_type = CATEGORICAL
            try:
                output_node = transformer.operate(input_node)
                nodes.append(output_node)
                print('Shape ==>', input_node.shape, output_node.shape)
                self.graph.add_node(output_node)
                self.graph.add_trans_in_graph(input_node, output_node, transformer)
            except MemoryError as e:
                print(e)

        transformer = Merger()
        output_node = transformer.operate(nodes)
        self.graph.add_node(output_node)
        self.graph.add_trans_in_graph(nodes, output_node, transformer)

        input_node = output_node
        transformer = VarianceSelector()
        output_node = transformer.operate(input_node)
        self.graph.add_node(output_node)
        self.graph.add_trans_in_graph(input_node, output_node, transformer)

        # PCA Dimension reduction.
        dim_reduction = False
        data_shape = output_node.shape
        if data_shape[1] >= 3 * data_shape[0]:
            dim_reduction = True
        if dim_reduction:
            input_node = output_node
            transformer = PcaDecomposer()
            output_node = transformer.operate(input_node)
            print('Shape ==>', input_node.shape, output_node.shape)
            self.graph.add_node(output_node)
            self.graph.add_trans_in_graph(input_node, output_node, transformer)

        self.incumbent = output_node
        return self.incumbent