import logging
import re
import numpy as np
import flwr as fl
import os
import janus_swi as janus
from popper.util import Settings, Stats, load_kbpath, Literal, format_rule
from popper.tester import Tester
from popper.loop import learn_solution
from typing import List, Tuple, Set, Dict

# Logging setup
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Dataset paths
kbpath = "complication2"
bk_file, ex_file, bias_file = load_kbpath(kbpath)

settings = Settings(cmd_line=False, kbpath=kbpath, ex_file=ex_file, bk_file=bk_file, bias_file=bias_file)
stats = Stats()
tester = Tester(settings)

# Test set constants
TEST_FOLDER = "testcom2"
BK_FILE = os.path.join(TEST_FOLDER, "bk.pl")
EXS_FILE = os.path.join(TEST_FOLDER, "exs.pl")


def load_rules(path):
    with open(path) as f:
        return f.read()
# Utilities

def extract_truths(path):
    pos, neg = set(), set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("pos("):
                m = re.match(r"pos\(complication\(([^)]+)\)\)\.", line)
                if m: pos.add(m.group(1))
            elif line.startswith("neg("):
                m = re.match(r"neg\(complication\(([^)]+)\)\)\.", line)
                if m: neg.add(m.group(1))
    return pos, neg

def clean_rule(rule: str) -> str:
    rule = rule.strip()
    return rule if rule.endswith('.') else rule + '.'

def load_received_rules() -> List[List[str]]:
    with open("received_hyp.pl") as f:
        return [[line.strip() for line in f if line.strip()]]
def clean_singletons(rules: List[str]) -> str:
    # Remplace toutes les occurrences de V1, V2... qui apparaissent une seule fois par '_'
    cleaned = []
    for rule in rules:
        vars_used = re.findall(r'\b(V[0-9]+)\b', rule)
        for var in set(vars_used):
            if vars_used.count(var) == 1:
                rule = re.sub(rf'\b{var}\b', '_', rule)
        if not rule.strip().endswith('.'):
            rule += '.'
        cleaned.append(rule)
    return "\n".join(cleaned)
    
def normalize_rule(rule: str) -> str:
    """Ensure Prolog rule ends with one and only one period."""
    rule = rule.strip()
    if rule.endswith(".."):
        rule = rule[:-1]
    if not rule.endswith("."):
        rule += "."
    return rule
def evaluate_hypotheses(hypotheses, bk_file, exs_file) -> Tuple[float, int, Dict[str, float]]:
    log.info("Evaluating from saved rules")
    pos_patients, neg_patients = extract_truths(exs_file)
    all_patients = sorted(pos_patients | neg_patients)
    true_labels = {p: 1 for p in pos_patients}
    true_labels.update({p: 0 for p in neg_patients})
    vote_counts = {p: 0 for p in all_patients}

    for i, hyp in enumerate(hypotheses):
        print(f"\n🔍 Testing H{i+1} with {len(hyp)} rules:")
        for r in hyp:
            print(f"   ▸ {r}")

        try:
            janus.query_once("retractall(complication(_))")
            janus.query_once("retractall(pos(_))")
            janus.query_once("retractall(neg(_))")

            janus.consult(bk_file)

            # Add point only if not already there
            prolog_code = "\n".join(r if r.strip().endswith('.') else r.strip() + '.' for r in hyp)
            print("\nprolog code:\n", prolog_code)
            janus.consult("prog", prolog_code)

            janus.consult(exs_file)

            predicted = {
                p for p in all_patients
                if janus.query_once("complication(P)", {"P": p}).get("truth")
            }
            for p in predicted:
                vote_counts[p] += 1

        except Exception as e:
            log.error(f"Erreur lors de l'injection: {e}")

    num_hypotheses = len(hypotheses)
    majority_threshold = num_hypotheses // 2 + 1
    final_preds = {p: 1 if vote_counts[p] >= majority_threshold else 0 for p in all_patients}

    tp = fn = fp = tn = 0
    for p in all_patients:
        pred, true = final_preds[p], true_labels[p]
        tp += pred == true == 1
        tn += pred == true == 0
        fp += pred == 1 and true == 0
        fn += pred == 0 and true == 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    print("\n✨ Métriques Globales :")
    print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

    return 1 - accuracy, len(all_patients), {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

def parse_literal(s):
    m = re.match(r"(\\w+)\\(([^()]*)\\)", s)
    if not m:
        log.warning(f"\u26a0\ufe0f Littéral mal formé ignoré: '{s}'")
        return None
    pred = m.group(1)
    args = tuple(int(a[1:]) if a.startswith("V") else a for a in m.group(2).split(","))
    return Literal(pred, args)
def parse_rule(rule_str):
    if ":-" not in rule_str:
        return parse_literal(rule_str), frozenset()
    head_str, body_str = rule_str.split(":-")
    head = parse_literal(head_str.strip())
    body_literals = re.findall(r"\\w+\\([^()]*\\)", body_str.strip())
    body = frozenset(filter(None, (parse_literal(lit) for lit in body_literals)))
    return head, body
def debug_rules():
    """
    Affiche les règles compilées dans le module Prolog 'prog' concernant 'complication/1'.
    """
    try:
        rules = list(janus.query("prog:clause(complication(H), B)"))
        print(f"📌 {len(rules)} règle(s) actuellement dans prog:")
        for r in rules:
            # Certaines règles peuvent ne pas avoir de corps (ex: faits)
            head = r.get("H", "?")
            body = r.get("B", "true")
            print(f" - complication({head}) :- {body}.")
    except Exception as e:
        print(f"⚠️ Erreur pendant debug_rules: {e}")
def build_prolog_code(hypothesis_rules):
    dynamic_decl = ":- dynamic complication/1."
    cleaned_rules = "\n".join(r.strip() for r in hypothesis_rules if r.strip())
    return f"{dynamic_decl}\n{cleaned_rules}"

def fix_singletons(rule):
    # crude but helpful in cleaning up _unused_ second arguments
    return rule.replace(",V1)", ",_)")



class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.settings = settings
        self.tester = tester
        self.stats = stats
        self.current_rules = []
        self.received_hypotheses = []

    def get_parameters(self, config):
        if not self.current_rules:
            return [np.array([])]
        rules_str = [format_rule(r) for r in self.current_rules]
        return [np.array(rules_str, dtype='<U1000')]

    def set_parameters(self, parameters):
        """
        Reconstruit une liste d’hypothèses à partir d’un tableau numpy contenant des règles à plat,
        séparées par '### HYP ###'.
        """
        if not parameters or len(parameters[0]) == 0:
            log.debug( "⚠️ Paramètres vides reçus, remise à zéro des hypothèses.")
            self.received_hypotheses = []
            return

        rule_list = parameters[0].tolist()
        cleaned_rules = [r.strip() for r in rule_list if r.strip()]

        hypotheses = []
        current_hypothesis = []

        for rule in cleaned_rules:
            if rule == "### HYP ###":
                if current_hypothesis:
                    hypotheses.append(current_hypothesis)
                    current_hypothesis = []
            else:
                current_hypothesis.append(rule)

        if current_hypothesis:
            hypotheses.append(current_hypothesis)

        self.received_hypotheses = hypotheses
        log.debug( f"📥 {len(hypotheses)} hypothèses reçues.")



    def fit(self, parameters, config):
        log.info("🔥 Apprentissage ILP...")
        prog, score, stats = learn_solution(self.settings)
        if not prog:
            self.current_rules = []
            return [np.array([])], 0, {}

        self.current_rules = prog
        return self.get_parameters(config), len(prog), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        metrics_per_hypothesis = []
        per_hypothesis_preds = []

        for i, hyp in enumerate(self.received_hypotheses):
            log.info(f"🔍 Testing H{i+1} with {len(hyp)} rules:")
            for r in hyp:
                log.info(f"   ▸ {r}")

            janus.query_once("retractall(complication(_))")

            janus.consult('testcom2/bk.pl')
            janus.consult('testcom2/exs.pl')

            def fix_singletons(rule):
                return rule.replace(",V1)", ",_)")

            prolog_code = "\n".join(
                fix_singletons(r.strip().rstrip(".")) + "." for r in hyp if r.strip()
            )
            prolog_code = ":- dynamic complication/1.\n" + prolog_code

            print("📤 Injecting this Prolog code:\n", prolog_code)

            try:
                janus.consult("prog", prolog_code)
            except Exception as e:
                log.error(f"❌ Error while injecting rules: {e}")

            any_rule = janus.query_once("prog:clause(complication(_),_)")
            print(f"✅ Au moins une règle ? {any_rule['truth']}")

            pos_patients, neg_patients = extract_truths(EXS_FILE)
            all_patients = sorted(list(pos_patients | neg_patients))

            print(" Patients dans EXS:", all_patients)
            print("PAtients positifs:", pos_patients)
            print("Patients negatifs:", neg_patients)

            preds = {}
            for p in all_patients:
                q = janus.query_once("complication(P)", {"P": p})
                preds[p] = q.get("truth", False)
            per_hypothesis_preds.append(preds)

        vote_counts = {p: 0 for p in all_patients}
        for preds in per_hypothesis_preds:
            for p, val in preds.items():
                if val:
                    vote_counts[p] += 1

        majority_threshold = len(per_hypothesis_preds) // 2 + 1
        final_preds = {}
        for p in all_patients:
            yes = vote_counts[p]
            no = len(per_hypothesis_preds) - yes
            final = yes >= majority_threshold
            final_preds[p] = final
            print(f"  - {p}: YES={yes}, NO={no} => Final: {'✅ YES' if final else '❌ NO'}")

        tp = tn = fp = fn = 0
        for p in all_patients:
            pred = final_preds[p]
            true = p in pos_patients
            if pred and true:
                tp += 1
            elif not pred and not true:
                tn += 1
            elif pred and not true:
                fp += 1
            elif not pred and true:
                fn += 1

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

        log.info(f"\n✨ Vote Majoritaire => Acc: {accuracy:.2f}, Prec: {precision:.2f}, Rec: {recall:.2f}")
        print(f"\n✉️ TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        best = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }

        return 1 - accuracy, len(all_patients), best


# Start the Flower client
fl.client.start_client(
    server_address="localhost:8080",
    client=FlowerClient().to_client()
)
