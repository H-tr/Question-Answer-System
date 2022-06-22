def get_robust_prediction(example, tokenizer, nbest=10, null_threshold=1.0):
    
    inputs = get_qa_inputs(example, tokenizer)
    start_logits, end_logits = model(**inputs)

    # get sensible preliminary predictions, sorted by score
    prelim_preds = preliminary_predictions(start_logits, 
                                           end_logits, 
                                           inputs['input_ids'],
                                           nbest)
    
    # narrow that down to the top nbest predictions
    nbest_preds = best_predictions(prelim_preds, nbest, tokenizer)

    # compute the probability of each prediction - nice but not necessary
    probabilities = prediction_probabilities(nbest_preds)
        
    # compute score difference
    score_difference = compute_score_difference(nbest_preds)

    # if score difference > threshold, return the null answer
    if score_difference > null_threshold:
        return "", probabilities[-1]
    else:
        return nbest_preds[0].text, probabilities[0]


# ----------------- Helper functions for get_robust_prediction ----------------- #
def get_qa_inputs(example, tokenizer):
    # load the example, convert to inputs, get model outputs
    question = example.question_text
    context = example.context_text
    return tokenizer.encode_plus(question, context, return_tensors='pt')

def get_clean_text(tokens, tokenizer):
    text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(tokens)
        )
    # Clean whitespace
    text = text.strip()
    text = " ".join(text.split())
    return text

def prediction_probabilities(predictions):

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    all_scores = [pred.start_logit+pred.end_logit for pred in predictions] 
    return softmax(np.array(all_scores))

def preliminary_predictions(start_logits, end_logits, input_ids, nbest):
    # convert tensors to lists
    start_logits = to_list(start_logits)[0]
    end_logits = to_list(end_logits)[0]
    tokens = to_list(input_ids)[0]

    # sort our start and end logits from largest to smallest, keeping track of the index
    start_idx_and_logit = sorted(enumerate(start_logits), key=lambda x: x[1], reverse=True)
    end_idx_and_logit = sorted(enumerate(end_logits), key=lambda x: x[1], reverse=True)
    
    start_indexes = [idx for idx, logit in start_idx_and_logit[:nbest]]
    end_indexes = [idx for idx, logit in end_idx_and_logit[:nbest]]

    # question tokens are between the CLS token (101, at position 0) and first SEP (102) token 
    question_indexes = [i+1 for i, token in enumerate(tokens[1:tokens.index(102)])]

    # keep track of all preliminary predictions
    PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
    )
    prelim_preds = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # throw out invalid predictions
            if start_index in question_indexes:
                continue
            if end_index in question_indexes:
                continue
            if end_index < start_index:
                continue
            prelim_preds.append(
                PrelimPrediction(
                    start_index = start_index,
                    end_index = end_index,
                    start_logit = start_logits[start_index],
                    end_logit = end_logits[end_index]
                )
            )
    # sort prelim_preds in descending score order
    prelim_preds = sorted(prelim_preds, key=lambda x: (x.start_logit + x.end_logit), reverse=True)
    return prelim_preds

def best_predictions(prelim_preds, nbest, tokenizer):
    # keep track of all best predictions

    # This will be the pool from which answer probabilities are computed 
    BestPrediction = collections.namedtuple(
        "BestPrediction", ["text", "start_logit", "end_logit"]
    )
    nbest_predictions = []
    seen_predictions = []
    for pred in prelim_preds:
        if len(nbest_predictions) >= nbest: 
            break
        if pred.start_index > 0: # non-null answers have start_index > 0

            toks = tokens[pred.start_index : pred.end_index+1]
            text = get_clean_text(toks, tokenizer)

            # if this text has been seen already - skip it
            if text in seen_predictions:
                continue

            # flag text as being seen
            seen_predictions.append(text) 

            # add this text to a pruned list of the top nbest predictions
            nbest_predictions.append(
                BestPrediction(
                    text=text, 
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit
                    )
                )
        
    # Add the null prediction
    nbest_predictions.append(
        BestPrediction(
            text="", 
            start_logit=start_logits[0], 
            end_logit=end_logits[0]
            )
        )
    return nbest_predictions

def compute_score_difference(predictions):
    """ Assumes that the null answer is always the last prediction """
    score_null = predictions[-1].start_logit + predictions[-1].end_logit
    score_non_null = predictions[0].start_logit + predictions[0].end_logit
    return score_null - score_non_null
