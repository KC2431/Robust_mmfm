import torch
from vlm_eval.attacks.attack import Attack
import math

class AFW(Attack):



    def __init__(self, 
                 model, 
                 targeted=False, 
                 mu=1e-22, 
                 mask_out='none', 
                 img_range=(0,1), 
                 steps=100, 
                 p=1.5, 
                 ver=False):
        """_summary_
        Implementation of the Auto Frank-Wolfe attack: https://arxiv.org/pdf/2205.07972


        Args:
            model (_type_): _description_
            targeted (bool, optional): _description_. Defaults to False.
            mu (int, optional): _description_. Defaults to 1.
            mask_out (str, optional): _description_. Defaults to 'none'.
            img_range (tuple, optional): _description_. Defaults to (0,1).
            steps (int, optional): _description_. Defaults to 100.
            p (float, optional): _description_. Defaults to 1.5.
        """

        super().__init__(model, targeted=targeted, img_range=img_range)
        self.steps = steps
        self.mu = mu
        self.p = p
        self.ver = ver
        if mask_out != 'none':
            self.mask_out = mask_out
        else:
            self.mask_out = None

    def _set_mask(self, data):
        mask = torch.ones_like(data)
        if self.mask_out == 'context':
            mask[:, :-1, ...] = 0
        elif self.mask_out == 'query':
            mask[:, -1, ...] = 0
        elif isinstance(self.mask_out, int):
            mask[:, self.mask_out, ...] = 0
        elif self.mask_out is None:
            pass
        else:
            raise NotImplementedError(f'Unknown mask_out: {self.mask_out}')
        return mask

    def argmax_soln(self, x, grad):

        gamma = torch.max(-x * torch.sign(grad), (1 - x) * torch.sign(grad))
        w_p_mu = torch.pow(torch.abs(grad),1 / (self.p - 1)) / (self.p * self.mu) ** (self.p - 1)
        delta = torch.min(gamma, w_p_mu) * torch.sign(grad)
        return delta

    def __call__(self, x):

        for param in self.model.model.parameters():
            param.requires_grad = False

        mask_out = self._set_mask(x)
        M = 2
        x_adv = (x + self.argmax_soln(x, torch.randn_like(x).to(self.device))).detach()
        x_adv = x_adv.clamp(*self.img_range)
        f_values = []

        for step in range(self.steps):
            
            x_adv.requires_grad = True
            loss = self.model(x_adv).sum() if not self.targeted else -self.model(x_adv).sum()
            loss.backward()
            f_values.append(loss.item())
            x_adv_grad = x_adv.grad.data * mask_out
            
            with torch.no_grad():
                M = 1e-1#self.cal_M(M=M, step=step, obj_values=f_values)
                lr = M / (2 + math.sqrt(step))
                s = x + self.argmax_soln(x=x, grad=x_adv_grad) * mask_out
                x_adv = ((1 - lr) * x_adv + lr * s) * mask_out
                #x_adv = torch.clamp(x_adv,*self.img_range)

                if self.ver and step % 20 == 0:
                    print(f"Iter: {step}, Loss: {loss}")
        pert = x_adv - x
        pert = pert * mask_out
        breakpoint() 
        return (x + pert * mask_out).detach()

    """
    def cal_M(self,M, step):
        return M * 0.75 if step > 10 and step % 10 == 0 else M
    """

    def cal_M(self, M, step, obj_values, checkpoint_interval=10, rho=0.75):
        """
        Function to update step size M based on two conditions.
        
        Args:
            M (float): Current step size.
            step (int): Current iteration step.
            obj_values (list): List of objective function values for each step.
            checkpoint_interval (int): Number of iterations between checkpoints.
            rho (float): Threshold parameter, defaults to 0.75.
            
        Returns:
            float: Updated step size M.
        """
    # Checkpoint condition
        if step > 0 and step % checkpoint_interval == 0:
            # Condition 1: Check if there has been a significant improvement
            start = max(0, step - checkpoint_interval)  # Avoid negative index
            end = step
            improvements = 0
            
            for i in range(start + 1, end):
                if obj_values[i] < rho * obj_values[start]:
                    improvements += 1

            # If at least 75% of the steps show improvement, proceed without halving M
            if improvements / checkpoint_interval < rho:
                M = M * 0.75  # Condition 1 triggered, halve the step size

            # Condition 2: Check if current objective value is close to max found so far
            if obj_values[step-1] >= max(obj_values[:step]):
                M = M * 0.75  # Condition 2 triggered, halve the step size

        return M
