from antlr4 import CommonTokenStream, InputStream, ParserRuleContext
from antlr4.error.ErrorListener import ErrorListener

from project.GraphQueryLexer import GraphQueryLexer
from project.GraphQueryParser import GraphQueryParser
from project.GraphQueryVisitor import GraphQueryVisitor


def program_to_tree(program: str) -> tuple[ParserRuleContext, bool]:
    input_stream = InputStream(program)
    lexer = GraphQueryLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = GraphQueryParser(stream)

    return parser.prog(), parser.getNumberOfSyntaxErrors() == 0


def nodes_count(tree: ParserRuleContext) -> int:
    class NodesCountVisitor(GraphQueryVisitor):
        def visitChildren(self, node):
            result = 1
            n = node.getChildCount()
            for i in range(n):
                c = node.getChild(i)
                child_result = c.accept(self)

                result += 1 if child_result is None else child_result

            return result

        def visitTerminal(self, node):
            return 1

    nodes_count_visitor = NodesCountVisitor()
    return nodes_count_visitor.visit(tree)


def tree_to_program(tree: ParserRuleContext) -> str:
    class ConstructTreeVisitor(GraphQueryVisitor):
        def defaultResult(self):
            return ""

        def aggregateResult(self, aggregate, nextResult):
            return f"{aggregate} {nextResult}".strip()

        def visitTerminal(self, node):
            return node.getText()

    construct_tree_visitor = ConstructTreeVisitor()
    return construct_tree_visitor.visit(tree)
