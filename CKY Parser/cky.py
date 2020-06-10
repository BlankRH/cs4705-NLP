"""
COMS W4705 - Natural Language Processing - Summer 19 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        pi = {}
        n = len(tokens)

        #init
        for i in range(0, n):
            for j in range(i+1, n+1):
                pi[(i,j)] = []
            for rule in self.grammar.rhs_to_rules[(tokens[i],)]:
                pi[(i, i+1)].append(rule[0])

        for l in range(2, n+1):
            for i in range(0, n-l+1):
                j = i + l
                for k in range(i+1, j):
                    for B in pi[(i, k)]:
                        for C in pi[(k, j)]:
                            for rule in self.grammar.rhs_to_rules[(B, C)]:
                                if rule[0] not in pi[(i, j)]:
                                    pi[(i, j)].append(rule[0])

        return 'S' in pi[(0, n)] or self.grammar.startsymbol in pi[(0, n)]

       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table= {}
        probs = {}
        n = len(tokens)

        #init
        for i in range(0, n):
            for j in range(i+1, n+1):
                table[(i,j)] = {}
                probs[(i,j)] = {}
            for rule in self.grammar.rhs_to_rules[(tokens[i],)]:
                table[(i, i+1)][rule[0]] = tokens[i]
                probs[(i, i+1)][rule[0]] = math.log(rule[2])

        for l in range(2, n+1):
            for i in range(0, n-l+1):
                j = i + l
                for k in range(i+1, j):
                    for B in table[(i, k)]:
                        for C in table[(k, j)]:
                            for rule in self.grammar.rhs_to_rules[(B, C)]:
                                prob = math.log(rule[2]) + probs[(i, k)][B] + probs[(k, j)][C]
                                if rule[0] not in table[(i, j)]:
                                    table[(i, j)][rule[0]] = ((B, i, k), (C, k, j))
                                    probs[(i, j)][rule[0]] = prob
                                elif probs[(i, j)][rule[0]] < prob:
                                    table[(i, j)][rule[0]] = ((B, i, k), (C, k, j))
                                    probs[(i, j)][rule[0]] = prob
                                #if rule[0] == 'NP' and i == 0 and j == 6:
                                 #   print(B, C, probs[(i, j)][rule[0]])
                                  #  print(prob)

        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4

    #if nt not in chart[(i, j)]:
     #   return ()

    if j == i+1:
        return (nt, chart[(i, j)][nt])

    left = chart[(i,j)][nt][0]
    right = chart[(i,j)][nt][1]
    tree = (nt, get_tree(chart, left[1], left[2], left[0]), get_tree(chart, right[1], right[2], right[0]))
 
    return tree 

def print_chart(table, n):

     for i in range(0, n):
            for j in range(i+1, n+1):
                print(table[(i, j)], end=' ')
            print()
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =  ['flights', 'between', 'tampa', 'and', 'saint', 'louis', '.']
        #toks =['miami', 'flights','cleveland', 'from', 'to','.']
        #print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        print_chart(table, len(toks))
        print(get_tree(table, 0, len(toks), grammar.startsymbol))

        
