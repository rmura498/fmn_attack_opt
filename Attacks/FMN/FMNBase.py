from Attacks.Attack import Attack


class FMNBase(Attack):

    def __init__(self,
                 model: nn.Module,
                 inputs: Tensor,
                 labels: Tensor,
                 norm: float,
                 targeted: bool = False,
                 steps: int = 10,
                 alpha_init: float = 1.0,
                 alpha_final: Optional[float] = None,
                 gamma_init: float = 0.05,
                 gamma_final: float = 0.001,
                 starting_points: Optional[Tensor] = None,
                 binary_search_steps: int = 10
                 ):
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.norm: norm
        self.targeted = targeted
        self.steps = steps
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.starting_points = starting_points
        self.binary_search_steps = binary_search_steps

    def _boundary_search(self):
        if self.starting_points is not None:
            is_adv = self.model(self.starting_points).argmax(dim=1)
            if not is_adv.all():
                raise ValueError('Starting points are not all adversarial.')
            lower_bound = torch.zeros(batch_size, device=device)
            upper_bound = torch.ones(batch_size, device=device)
            for _ in range(binary_search_steps):
                epsilon = (lower_bound + upper_bound) / 2
                mid_points = mid_point(x0=inputs, x1=starting_points, epsilon=epsilon)
                pred_labels = model(mid_points).argmax(dim=1)
                is_adv = (pred_labels == labels) if targeted else (pred_labels != labels)
                lower_bound = torch.where(is_adv, lower_bound, epsilon)
                upper_bound = torch.where(is_adv, epsilon, upper_bound)

            delta = mid_point(x0=inputs, x1=starting_points, epsilon=epsilon) - inputs
        else:
            delta = torch.zeros_like(inputs)

    def _init_attack(self):
        pass

    def run(self):
        pass