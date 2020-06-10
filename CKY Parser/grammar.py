"""
COMS W4705 - Natural Language Processing - Summer 19 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1

        for rules in self.lhs_to_rules:
            prob = 0.0
            for rule in self.lhs_to_rules[rules]:
                prob = fsum([prob, rule[2]])
                if len(rule[1]) == 2 and not rule[1][0].isupper() and rule[1][1].isupper():
                    print(rule)
                    return False
                if len(rule[1]) == 1 and rule[1][0].isupper():
                    print(rule)
                    return False
                if len(rule[1]) != 2 and len(rule[1]) != 1:
                    print(rule)
                    return False
            if abs(prob - 1.0) > 1e-11:
                print(prob)
                print(self.lhs_to_rules[rules])
                return False


        return True


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)
    print(grammar.verify_grammar())
        
