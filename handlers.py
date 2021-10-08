import re

from abc import ABC, abstractmethod

CLAUSAL_DEP_TAGS = [
    'ROOT',
    'csubj', 'ccomp', 'xcomp',
    'advcl', 'acl'
]

DEP_CLAUSE_DEP_TAGS = [
    'csubj', 'ccomp', 'xcomp',
    'advcl', 'acl'
]

class RuleHandler(ABC):
    def __init__(self, lang_model):
        self.model = lang_model
    
    @abstractmethod
    def check_tag(self, err_tag: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def assign_class(self, sentence: str, err_span: str, correction: str, err_tag: str) -> int:
        raise NotImplementedError
    
    def get_err_sent(self, sentence: str) -> str:
        return sentence.replace('<b>','').replace('</b>','')
    
    def get_corr_sent(self, sentence: str, correction: str) -> str:
        return re.sub('<b>.+?</b>', correction, sentence)
    
    def get_root(self, string: str):
        parse = self.model(string)
        for token in parse:
            if token.dep_ == 'ROOT':
                return token
        return None
    
    def get_corr_root(self, sent: str, correction: str):
        left_length = len(self.model(sent[:sent.find('<b>')]))
        corr_root_i = self.get_root(correction).i + left_length
        corr_sent = self.get_corr_sent(sent, correction)
        corr_root_node = self.model(corr_sent)[corr_root_i]
        return corr_root_node
    
    def get_min_err_clause(self, sentence: str, correction: str):
        # Возвращает минимальную клаузу,
        # целиком включающую область ошибки
        # как узел дерева spacy (spacy.token)

        corr_root_node = self.get_corr_root(sentence, correction)
        
        head = corr_root_node
        while head.dep_ not in CLAUSAL_DEP_TAGS:
            if head.dep_ == 'conj' and head.head.dep_ in CLAUSAL_DEP_TAGS:
                break
            head = head.head
        
        return head
    
    def __call__(self, sentence: str, err_span: str, correction: str, err_tag: str) -> int:
        if self.check_tag(err_tag):
            return self.assign_class(sentence, err_span, correction, err_tag)
        return 0

# Пункт 1:
class AgreementHandler(RuleHandler):
    def check_tag(self, err_tag: str) -> bool:
        if err_tag == 'Agreement_errors':
            return True
        return False
    
    def assign_class(self, sentence: str, err_span: str, correction: str, err_tag: str) -> int:
        # TO DO: Как-то предусмотреть два варианта исправления, если возможно
        try:
            corr_sent = self.get_corr_sent(sentence, correction)
        except TypeError:
            print(f"'{sentence}'", f"'{correction}'")
        
        parse = self.model(sentence)

        # если в составе подлежащего есть:
        ## (это если ошибка именно на подлежащем или нет?)
        min_err_clause = self.get_min_err_clause(sentence, correction)
        subject = [child for child in min_err_clause.children if child.dep_ in ('nsubj','csubj')]

        if subject:
            subject = subject[0]

            ## Latest improvement
            if subject.dep_ == "csubj":
                return 3
            ## End of latest improvement

            # Если подлежащее - collective noun, уровень 3 (в стандарте UD это есть, но парсер SpaCy при мне такого не обнаруживал)
            if "Number=Coll" in subject.morph:
                return 3
            
            # если подлежащее и сказуемое в разных клаузах ИЛИ если подлежащее и сказуемое разделены вставленной клаузой (или фразой из 5 слов и больше), то уровень 3
            ## если подлежащее и сказуемое разделены фразой из 5 слов или больше
            if min_err_clause.i > subject.i:
                span = parse[subject.i+1:min_err_clause.i]
            elif subject.i > min_err_clause.i:
                span = parse[min_err_clause.i+1:subject.i]
            
            if len([token for token in span if token.pos_ != 'PUNCT']) >= 5:
                return 3
            
            # проверим, что в этом span'е существует целая клауза - достаточно того,
            ## чтобы был хотя бы один клаузальный тэг:
            for token in span:
                if token.dep_ in CLAUSAL_DEP_TAGS:
                    return 3

            for child in subject.children:
                ## любая предложная конструкция
                if child.pos == 'ADP':
                    return 2
                ## любое сочинение (and//or//as well as)
                if child.dep_ == 'conj':
                    return 2
                ## clarifications or examples separated on either side by
                ## dashes, commas or brackets (or semicolon and dash)
                if child.dep_ == 'appos':
                    return 2
            ## если есть together with, along with, accompanied by etc.
            ## either or//neither nor//both (and)
            str_children = ' '.join([str(token) for token in subject.children])
            if 'together with' in str_children or\
            'along with' in str_children or\
            'accompanied by' in str_children or\
            'either' in str_children or 'both' in str_children:
                return 2
            
            # если в составе сказуемого есть:
            for child in min_err_clause.children:
                ## любое сочинение
                if child.dep_ == 'conj':
                    return 2
        
        return 1

# Пункты 2, 4, 5, 6, 8, 11, 13, 14, 15, 17, 19, 20
# (Про 14 и 15 пункты уточнить, что они попадают сюда)
class TagOnlyHandler(RuleHandler):
    ## прописать все соотв. тэги
    lvl1_tags = ['Voice', 'Spelling', 'Capitalisation', 'Noun_number']
    lvl2_tags = ['Comparative_adj', 'Superlative_adj', # degrees of adjectives/adverbs
    'Comparative_adv', 'Superlative_adv',
    'Modals', 'Tense_form',
    'Infinitive_constr','Gerund_phrase',
    'Countable_uncountable', 'Numerals']
    lvl3_tags = ['Prepositions', # prepositional tags
    'Prepositional_noun', 'Prepositional_adjective',
    'Prepositional_verb', 'Prepositional_adv',
    "Verb_pattern", # verb pattern errors
    "Intransitive",
    "Transitive",
    "Reflexive_verb",
    "Presentation",
    "Ambitransitive",
    "Two_in_a_row",
    "Verb_Inf",
    "Verb_Gerund",
    "Verb_Inf_Gerund",
    "Verb_Bare_Inf",
    "Verb_object_bare",
    "Restoration_alter",
    "Verb_part",
    "Get_part",
    "Complex_obj",
    "Verbal_idiom",
    "Prepositional_verb",
    "Dative",
    "Followed_by_a_clause",
    "that_clause",
    "if_whether_clause",
    "that_subj_clause",
    "it_conj_clause",
    "Discourse", # discourse tags
    "Ref_device",
    "Coherence",
    "Linking_device",
    "Inappropriate_register",
    "Absence_comp_sent",
    "Redundant_comp",
    "Absence_explanation"]
    all_tags = lvl1_tags + lvl2_tags + lvl3_tags

    def check_tag(self, err_tag: str) -> bool:
        if err_tag in self.all_tags:
            return True
        return False
    
    def assign_class(self, sentence: str, err_span: str, correction: str, err_tag: str) -> int:
        if err_tag in self.lvl1_tags:
            return 1
        elif err_tag in self.lvl2_tags:
            return 2
        elif err_tag in self.lvl3_tags:
            return 3

# Пункт 3
class LexicalErrorHandler(RuleHandler):
    tags = [
        'lex_item_choice',
        'lex_part_choice',
        'Word_choice',
        'Often_confused',
        'Absence_comp_colloc',
        'Redundant'
    ]

    def check_tag(self, err_tag: str) -> bool:
        if err_tag in self.tags:
            return True
        return False
    
    def assign_class(self, sentence: str, err_span: str, correction: str, err_tag: str) -> int:
        if len([i for i in self.model(err_span) if i.pos_ != 'PUNCT']) == 1:
            return 2
        return 3

# Пункт 7
class StandardWordOrderHandler(RuleHandler):
    def check_tag(self, err_tag: str) -> bool:
        if err_tag == 'Word_order':
            return True
        return False
    
    def assign_class(self, sentence: str, err_span: str, correction: str, err_tag: str) -> int:
        if len([i for i in self.model(err_span) if i.pos_ != 'PUNCT']) <= 3:
            return 2
        return 3

# Пункт 8, 21
# class NumeralsHandler(RuleHandler):
#     def check_tag(self, err_tag: str) -> bool:
#         if err_tag == "Numerals":
#             return True
#         return False
    
#     def assign_class(self, sentence: str, err_span: str, correction: str, err_tag: str) -> int:
#         err_span_split = self.model(err_span)

#         # если в состав области ошибки входят числительные - 2:
#         if 'NUM' in [token.pos_ for token in err_span_split]:
#             return 2
        
#         # ошибки в числе существительного после числа - 1:
#         if self.get_root(err_span).pos_ == 'NOUN':
#             return 1
        
#         return 0

# Пункт 9
class TenseChoiceHandler(RuleHandler):
    def check_tag(self, err_tag: str) -> bool:
        if err_tag == 'Tense_choice':
            return True
        return False
    
    def assign_class(self, sentence: str, err_span: str, correction: str, err_tag: str) -> int:
        # те, которые в предложениях больше чем с 2 клаузами
        corr_sent = self.get_corr_sent(sentence, correction)
        corr_sent_parsed = self.model(corr_sent)
        clause_count = 0

        for token in corr_sent_parsed:
            if token.dep_ in DEP_CLAUSE_DEP_TAGS:
                clause_count += 1
            if clause_count > 2:
                return 3
        # и те, в которых глагольная форма в исправлении больше 2 слов - 3:
        correction_parsed = self.model(correction)
        verb_count = 0

        for token in correction_parsed:
            if token.pos_ == 'VERB':
                verb_count += 1
            if verb_count > 2:
                return 3
        
        return 2

# Пункт 10
class DeterminersHandler(RuleHandler):
    tags = ['Articles','Determiners']

    def check_tag(self, err_tag: str) -> bool:
        if err_tag in self.tags:
            return True
        return False
    
    # спросить:
    def assign_class(self, sentence: str, err_span: str, correction: str, err_tag: str) -> int:
        # при именах собственных - 1
        correction_parsed = self.model(correction)
        for token in correction_parsed:
            if token.tag_ in ('NNP', 'NNPS'):
                return 1
        # c остальными существительными или вообще без существительного - 2
        # если артикль заменяется (неважно, с какого на какой) при существительном, которое может функционировать и как исчисляемое, и как неисчисляемое, то уровень 3
        return 2

# Пункт 12
class CategoryConfusionHandler(RuleHandler):
    tags = ["Category_confusion","Formational_affixes",
    "Suffix","Prefix"]
    
    def check_tag(self, err_tag: str) -> bool:
        if err_tag == "Category_confusion":
            return True
        return False
    
    def assign_class(self, sentence: str, err_span: str, correction: str, err_tag: str) -> int:
        # ошибки  Confusion of Categories – 2, за исключением тех, в которых часть речи в исправлении другая, нежели в области ошибки – эти 3 уровня
        err_span_parsed = self.model(err_span)
        correction_parsed = self.model(correction)
        if len(err_span_parsed)==len(correction_parsed)==1:
            if err_span_parsed[0].pos_ != correction_parsed[0].pos_:
                return 3
        return 2

# Пункт 18
class ZeroVerbNegationHandler(RuleHandler):
    # Tag-independent handler:
    def check_tag(self, err_tag: str) -> bool:
        return True

    def assign_class(self, sentence: str, err_span: str, correction: str, err_tag: str) -> int:
        # если в области ошибки глагол HASN’T //HAVEN’T, после которого нет основного глагола, но есть кванторное слово ANY, ANYBODY, ANYONE, ANYTHING, ANYWHERE – уровень 3
        if re.search('ha(s|ve)(n\'t| not) any', err_span.lower()):
            return 3
        return 0

# Пункт 22
# какой это тэг?
class DeterminerNounAgreement(RuleHandler):
    # Tag-independent handler:
    def check_tag(self, err_tag: str) -> bool:
        return True

    def assign_class(self, sentence: str, err_span: str, correction: str, err_tag: str) -> int:
        # согласование межлу determiner и существительным - уровень 1
        root = self.get_root(correction)
        if root.pos_ == "NOUN":
            children_deps = [child.dep for child in root.children]
            if "det" in children_deps:
                return 1
        return 0