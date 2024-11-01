#define DEBUG_MODE_FLAG true 

int main(){

    //Remember
    //first - linear is not numerically stable - this is to be researched
    //second - linear does not uniformly distribute the context - unless it is one dimensional x two dimensional linear opeartion
    //the drawbacks to one_dimensional x two_dimensional linear is that the possibility space is too big - this leads to remembering the result rather than actually intelligently connecting the result
    //that's why two_dimensional x two_dimensional linear is preferred - yet it does not uniformly distribute the context - this is bad
    //the balance is to be found
    //people tried to use transpose(0, 1) and rotary embeddings to actually solve that without realizing that 
    //uniform context distribution could be briefly described as given two random logits, the influence of a logit on another is uniformly distributed among all possible pairs after a certain number of tranformations
    //all there is to AGI is that - the dimensional reduction - the possiblity space - the uniform distribution - the numerical stable model - the dynamic transformation path
}