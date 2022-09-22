# Possible issues

- Is the update rule we use in `mc_prediction` really the correct one? (`V[state] = (returns_count[state] * V[state] + G) / (returns_count[state] + 1)`)
- Is setting the NaN values (unexplored states) to zero allowed?
- Is it ok to use `self.sample_action` in `SimpleBlackjackPolicy.get_probs`? ( I think it is, as we pass the codegrade test)
- Isn't too `RandomBlackjackPolicy` too simple?
- Is the update rule we use in `mc_importance_sampling` correct? (`V[state] = V[state] + 1 / (returns_count[state] + 1) * (W * G - V[state])`)
- Is the `if W == 0: break` statement correct in `mc_importance_sampling`?
- Is the `W = W * pi / b` correct in `mc_importance_sampling`?
- Maybe it isn't so surprising that we get bad results for the ordinary importance sampling, maybe off-policy ordinary iportance sampling just works really bad for the blackjack game.
