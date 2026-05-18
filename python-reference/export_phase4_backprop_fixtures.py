import json, torch
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'public'/'phase4-backprop'
out.mkdir(parents=True,exist_ok=True)
torch.manual_seed(0)
# tiny tensors
x=torch.randn(1,2,4,4,requires_grad=True)
w=torch.randn(3,2,3,3)
b=torch.randn(3)
t=torch.randn(1,2,4,4)
# relu
relu_in=torch.tensor([[[[-1.,2.],[3.,-4.]]]],requires_grad=True)
relu_out=torch.relu(relu_in); relu_out.backward(torch.tensor([[[[1.,5.],[2.,4.]]]]))
# pool
pool_in=torch.tensor([[[[1.,5.,3.,2.],[4.,0.,1.,7.],[2.,8.,6.,4.],[3.,1.,9.,0.]]]],requires_grad=True)
pool_out=torch.nn.functional.max_pool2d(pool_in,2,2)
pool_out.backward(torch.tensor([[[[1.,2.],[3.,4.]]]]))
# normalize
mean=torch.tensor([0.4,0.5]); std=torch.tensor([0.2,0.25])
norm_in=torch.randn(1,2,2,2,requires_grad=True)
norm=((norm_in-mean.view(1,-1,1,1))/std.view(1,-1,1,1)); norm.backward(torch.randn_like(norm_in))
# conv backward input
a=torch.randn(1,2,3,3,requires_grad=True)
weight=torch.randn(4,2,3,3)
conv=torch.nn.functional.conv2d(a,weight,bias=None,padding=1)
go=torch.randn_like(conv); conv.backward(go)
# gram backward
g_in=torch.randn(1,3,2,2,requires_grad=True)
a1,b1,c1,d1=g_in.shape
f=g_in.view(a1*b1,c1*d1); gram=f@f.t()/(a1*b1*c1*d1)
grad_gram=torch.randn_like(gram); gram.backward(grad_gram)
# content/style backward
ci=torch.randn(1,3,2,2,requires_grad=True); ct=torch.randn(1,3,2,2)
cl=torch.nn.functional.mse_loss(ci,ct); cl.backward()
si=torch.randn(1,3,2,2,requires_grad=True); st=torch.randn(1,3,2,2)
fs=si.view(3,4); gs=(fs@fs.t())/12
ft=st.view(3,4); gt=(ft@ft.t())/12
sl=torch.nn.functional.mse_loss(gs,gt); sl.backward()
# e2e tiny
inp=torch.randn(1,2,4,4,requires_grad=True)
style=torch.randn(1,2,4,4); content=torch.randn(1,2,4,4)
weight2=torch.randn(2,2,3,3)
outi=torch.nn.functional.max_pool2d(torch.relu(torch.nn.functional.conv2d(inp,weight2,padding=1)),2,2)
outs=torch.nn.functional.max_pool2d(torch.relu(torch.nn.functional.conv2d(style,weight2,padding=1)),2,2)
outc=torch.nn.functional.max_pool2d(torch.relu(torch.nn.functional.conv2d(content,weight2,padding=1)),2,2)
fi=outi.view(2,4); fs=outs.view(2,4)
loss=torch.nn.functional.mse_loss(outi,outc)+torch.nn.functional.mse_loss((fi@fi.t())/8,(fs@fs.t())/8)
loss.backward()
fixture={
'relu':{'input':[1,1,2,2],'inputValues':relu_in.detach().view(-1).tolist(),'gradOutValues':[1.,5.,2.,4.],'expectedGradIn':relu_in.grad.view(-1).tolist()},
'pool':{'input':[1,1,4,4],'inputValues':pool_in.detach().view(-1).tolist(),'gradOut':[1,1,2,2],'gradOutValues':[1.,2.,3.,4.],'expectedGradIn':pool_in.grad.view(-1).tolist()},
'normalize':{'shape':[1,2,2,2],'std':std.tolist(),'gradOut':norm.grad_fn is None and [],'gradOutValues':norm.grad if False else [],'inputValues':norm_in.detach().view(-1).tolist()},
'convBackwardInput':{'inputShape':[1,2,3,3],'weightShape':[4,2,3,3],'weightValues':weight.view(-1).tolist(),'gradOutShape':[1,4,3,3],'gradOutValues':go.view(-1).tolist(),'expectedGradIn':a.grad.view(-1).tolist()},
'gramBackward':{'inputShape':[1,3,2,2],'inputValues':g_in.detach().view(-1).tolist(),'gradOutShape':[1,1,3,3],'gradOutValues':grad_gram.view(-1).tolist(),'expectedGradIn':g_in.grad.view(-1).tolist()},
'contentLossBackward':{'shape':[1,3,2,2],'inputValues':ci.detach().view(-1).tolist(),'targetValues':ct.view(-1).tolist(),'expectedGradIn':ci.grad.view(-1).tolist()},
'styleLossBackward':{'shape':[1,3,2,2],'inputValues':si.detach().view(-1).tolist(),'targetValues':st.view(-1).tolist(),'expectedGradIn':si.grad.view(-1).tolist()},
'e2e':{'inputShape':[1,2,4,4],'inputValues':inp.detach().view(-1).tolist(),'styleValues':style.view(-1).tolist(),'contentValues':content.view(-1).tolist(),'weightShape':[2,2,3,3],'weightValues':weight2.view(-1).tolist(),'expectedInputGrad':inp.grad.view(-1).tolist()}
}
# normalize proper gradOut
norm_in2=torch.tensor(norm_in.detach(),requires_grad=True)
grad_out=torch.randn_like(norm_in2)
out_norm=(norm_in2-mean.view(1,-1,1,1))/std.view(1,-1,1,1)
out_norm.backward(grad_out)
fixture['normalize']['gradOutValues']=grad_out.view(-1).tolist(); fixture['normalize']['expectedGradIn']=norm_in2.grad.view(-1).tolist()
(out/'phase4_backprop_fixture.json').write_text(json.dumps(fixture))
print('wrote',out/'phase4_backprop_fixture.json')
