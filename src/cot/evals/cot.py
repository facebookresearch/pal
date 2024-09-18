"""
Evaluation metric

License
-------
This source code is licensed under the CC license found in the LICENSE file
in the root directory of this source tree.

@ 2024, Meta
"""

import torch

from cot.config import TOKEN_DICT


class AccuracyEval:

    def __init__(self, lengths):
        self.meaning = [f"acc_len_{leng}" for leng in lengths]
        self.eval_dim = len(self.meaning)

    def __call__(self, model, dataset, pred=None, logits=None):
        """
        Compute sequences accuracy

        Parameters
        ----------
        model: Transformer
            Model to compute scores.
        dataset: SequenceDataset
            Dataset to compute scores.
        pred: Torch.tensor
            Pre-computed prediction.
        logits: Torch.tensor
            Pre-computed logits.

        Returns
        -------
        errors_by_len: torch.Tensor of n_len
            Full sentence accuracy conditioned over sequence lengths.
        """
        device = list(model.parameters())[0].device
        data = dataset.data.to(device=device, dtype=torch.long)

        if pred is None:
            if logits is None:
                logits = model(data)
            pred = logits.argmax(dim=-1)

        pred = pred[:, :-1]
        ground_truth = data[:, 1:]

        ind = ground_truth == TOKEN_DICT["EoI"]
        cot_mask = ind.cumsum(axis=1)
        cot_mask[ind] = 0
        cot_mask = cot_mask.to(dtype=bool)
        pred[~cot_mask] = ground_truth[~cot_mask]

        token_errors = pred != ground_truth
        seq_errors = token_errors.any(dim=1)

        eois = torch.argmax((data == TOKEN_DICT["EoI"]).to(int), dim=1)
        all_eois = torch.unique(eois)

        errors_by_len = torch.empty((len(all_eois)), device=eois.device, dtype=float)

        # group and process sequences by lengths
        for i, eoi in enumerate(all_eois):
            ind = eois == eoi
            errors_by_len[i] = 1 - seq_errors[ind].to(float).mean()

        return errors_by_len


class GrammarEval:

    def __init__(self):
        self.meaning = ["boi", "eoi", "eos"]
        self.eval_dim = len(self.meaning)

    def __call__(self, model, dataset, pred=None, logits=None):
        """
        Compute sequences grammar correctness.

        Parameters
        ----------
        model: Transformer
            Model to compute scores.
        dataset: SequenceDataset
            Dataset to compute scores.
        pred: Torch.tensor
            Pre-computed prediction.
        logits: Torch.tensor
            Pre-computed logits.

        Returns
        -------
        out: torch.Tensor of size 6 * n_len
            Metrics evaluating if boi and eoi appears only once and if eos is a 'sinking' state.
        """
        device = list(model.parameters())[0].device

        if pred is None:
            if logits is None:
                data = dataset.data.to(device)
                logits = model(data)
            pred = logits.argmax(dim=-1)

        out = torch.zeros(3, device=device, dtype=float)
        out[0] = ((pred == TOKEN_DICT["BoS"]).sum(dim=-1) == 1).to(float).mean().item()
        out[1] = ((pred == TOKEN_DICT["EoI"]).sum(dim=-1) == 1).to(float).mean().item()
        tmp = (pred == TOKEN_DICT["EoS"]).to(int)
        tmp1 = tmp.sum(dim=-1)
        tmp1 *= -1
        tmp1 += tmp.shape[1]
        tmp2 = tmp.argmax(dim=-1)
        out[2] = (tmp1 == tmp2).to(float).mean()

        return out


class AttentionEval:

    def __init__(self, lengths):
        self.meaning = []
        for leng in lengths:
            self.meaning += [
                f"attn0_inv_{leng}",
                f"attn1_inv_{leng}",
                f"attn0_peaky_abs_{leng}",
                f"attn0_peaky_thres_{leng}",
                f"attn1_peaky_abs_{leng}",
                f"attn1_peaky_thres_{leng}",
            ]
        self.eval_dim = len(self.meaning)

    def __call__(self, model, dataset, attns=None):
        """
        Compute attention scores.

        Parameters
        ----------
        model: Transformer
            Model to compute scores.
        dataset: SequenceDataset
            Dataset to compute scores.
        attns: Torch.tensor
            Pre-computed attention maps.

        Returns
        -------
        out: torch.Tensor of size 6 * n_len
            Metrics evaluating attention heads peakiness and invariance to token.
        """
        device = list(model.parameters())[0].device
        data = dataset.data.to(device)

        if attns is None:
            _, attns = model(data, verbose=True)

        attn_inv, attn_peaky = self.attention_metrics(data, attns)
        return torch.hstack((attn_inv, attn_peaky)).flatten()

    @staticmethod
    def attention_metrics(sequences, attentions):
        """
        Compute success metrics to CoT emergence.

        Parameters
        ----------
        sequences: tensor of size (bsz, seq_len)
            Token sequences.
        attentions: tensore of size (n_layer=2, bsz, n_head=1, seq_len, seq_len)
            Attention maps.

        Returns
        -------
        attn_inv: tensor of size (len, n_layer * n_head = 2)
            Score of invarance of attention to token sequence.
        attn_peaky: tensore of size (len, 2 * n_layer * n_head = 4)
            Success metrics for the attention maps.
        """
        eois = torch.argmax((sequences == TOKEN_DICT["EoI"]).to(int), dim=1)
        all_eois = torch.unique(eois)

        attn_inv = torch.empty((len(all_eois), 2), device=eois.device, dtype=float)
        attn_peaky = torch.empty((len(all_eois), 4), device=eois.device, dtype=float)

        # group and process sequences by lengths
        for i, eoi in enumerate(all_eois):
            ind = eois == eoi

            # handcrafted EoS given EoI
            eos = 2 * eoi

            # handcrafted attention score to look at
            attn0 = attentions[0, ind, 0, eoi + 1 : eos, eoi]
            attn1 = torch.diagonal(attentions[1, ind, 0, eoi : eos - 1, 1:eoi], dim1=1, dim2=2)

            # how does attention change for different sequences
            attn_inv[i, 0] = 1 - attn0.std(dim=0).mean()
            attn_inv[i, 1] = 1 - attn1.std(dim=0).mean()

            # how much the attention is picky
            attn_peaky[i, 0] = attn0.mean()
            attn_peaky[i, 1] = (attn0 > 0.5).to(dtype=float).mean()
            attn_peaky[i, 2] = attn1.mean()
            attn_peaky[i, 3] = (attn1 > 0.5).to(dtype=float).mean()
        return attn_inv, attn_peaky


class FullEval:

    def __init__(self, lengths):
        self.acc_eval = AccuracyEval(lengths)
        self.grammar_eval = GrammarEval()
        self.attn_eval = AttentionEval(lengths)
        self.meaning = self.acc_eval.meaning + self.grammar_eval.meaning + self.attn_eval.meaning
        self.eval_dim = len(self.meaning)

    def __call__(self, model, dataset):
        device = list(model.parameters())[0].device
        data = dataset.data.to(device)
        logits, attns = model(data, verbose=True)
        pred = logits.argmax(dim=-1)

        acc_eval = self.acc_eval(model, dataset, pred=pred, logits=logits)
        grammar_eval = self.grammar_eval(model, dataset, pred=pred, logits=logits)
        attn_eval = self.attn_eval(model, dataset, attns=attns)
        out = torch.hstack((acc_eval, grammar_eval, attn_eval))
        return out
