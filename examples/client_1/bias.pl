max_clauses(6).
max_vars(5).
max_body(6). 
head_pred(complication,1).

body_pred(extubation,2).
body_pred(ic,1).
body_pred(moyensupraglottique,1).
body_pred(recentnecksurgery,2).
body_pred(asthma,1).
body_pred(thyromentaldistance,1).
body_pred(fixation_level,2).
body_pred(betweenhyoidboneandthyroid,1).
body_pred(copd,1).
body_pred(largetongue,1).
body_pred(diagnostic,2).
body_pred(cervicalcollar,2).
body_pred(hta,1).
body_pred(coronary_artery_disease,1).
body_pred(lemonscore,2).
body_pred(secondoperator,1).
body_pred(intubation_duration,2).
body_pred(eichman,1).
body_pred(age,2).
body_pred(high_arched_palate,1).
body_pred(elderly,1).
body_pred(indication,2).
body_pred(shortneck,1).
body_pred(openingmouth,1).
body_pred(hard_intubation,1).
body_pred(experience,1).
body_pred(diabetes,1).
body_pred(necktrauma,1).
body_pred(number_probes,2).
body_pred(other_conditions,2).
body_pred(selective_intubation,1).
body_pred(number_attempts,2).
body_pred(patientobese,1).
body_pred(cormack,2).
body_pred(hard_laryngo,1).
type(complication,(patient,)).

type(extubation,(patient,real)).
type(ic,(patient,)).
type(moyensupraglottique,(patient,)).
type(recentnecksurgery,(patient,real)).
type(asthma,(patient,)).
type(thyromentaldistance,(patient,)).
type(fixation_level,(patient,real)).
type(betweenhyoidboneandthyroid,(patient,)).
type(copd,(patient,)).
type(largetongue,(patient,)).
type(diagnostic,(patient,real)).
type(cervicalcollar,(patient,real)).
type(hta,(patient,)).
type(coronary_artery_disease,(patient,)).
type(lemonscore,(patient,real)).
type(secondoperator,(patient,)).
type(intubation_duration,(patient,real)).
type(eichman,(patient,)).
type(age,(patient,real)).
type(high_arched_palate,(patient,)).
type(elderly,(patient,)).
type(indication,(patient,real)).
type(shortneck,(patient,)).
type(openingmouth,(patient,)).
type(hard_intubation,(patient,)).
type(experience,(patient,)).
type(diabetes,(patient,)).
type(necktrauma,(patient,)).
type(number_probes,(patient,real)).
type(other_conditions,(patient,real)).
type(selective_intubation,(patient,)).
type(number_attempts,(patient,real)).
type(patientobese,(patient,)).
type(cormack,(patient,real)).
type(hard_laryngo,(patient,)).
direction(complication,(in,)).

direction(extubation,(in,out)).
direction(ic,(in,)).
direction(moyensupraglottique,(in,)).
direction(recentnecksurgery,(in,out)).
direction(asthma,(in,)).
direction(thyromentaldistance,(in,)).
direction(fixation_level,(in,out)).
direction(betweenhyoidboneandthyroid,(in,)).
direction(copd,(in,)).
direction(largetongue,(in,)).
direction(diagnostic,(in,out)).
direction(cervicalcollar,(in,out)).
direction(hta,(in,)).
direction(coronary_artery_disease,(in,)).
direction(lemonscore,(in,out)).
direction(secondoperator,(in,)).
direction(intubation_duration,(in,out)).
direction(eichman,(in,)).
direction(age,(in,out)).
direction(high_arched_palate,(in,)).
direction(elderly,(in,)).
direction(indication,(in,out)).
direction(shortneck,(in,)).
direction(openingmouth,(in,)).
direction(hard_intubation,(in,)).
direction(experience,(in,)).
direction(diabetes,(in,)).
direction(necktrauma,(in,)).
direction(number_probes,(in,out)).
direction(other_conditions,(in,out)).
direction(selective_intubation,(in,)).
direction(number_attempts,(in,out)).
direction(patientobese,(in,)).
direction(cormack,(in,out)).
direction(hard_laryngo,(in,)).
:- clause(C), #count{V : var_type(C,V,patient)} != 1.