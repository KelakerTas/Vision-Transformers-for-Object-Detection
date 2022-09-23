import torch

class PatchesCreate(torch.nn.Module):
    def __init__(self, patch_size):
        super(PatchesCreate, self).__init__()
        self.patch_size = patch_size
        
    def forward(self, images):
        
        batch_size = images.shape[0]
        image_size = images.shape[1]
        grid_size = int(image_size/self.patch_size)
        
        Patches_Mask = torch.zeros(batch_size, grid_size, grid_size, self.patch_size**2*3)
        
        for index, image in enumerate(images):
            for i in range(grid_size):
                for j in range(grid_size):
                    patch_image = image[i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size, :]
                    Patches_Mask[index, i, j, : ] = torch.flatten(patch_image)
                    Patches = torch.reshape(Patches_Mask, (batch_size, -1, self.patch_size**2*3))
                    
        #Patches = Patches.to(device)
        return Patches
    
    
class EncodePatches(torch.nn.Module):
    def __init__(self, num_patches, project_dim, patch_dim):
        
        ##Interesting !!! it has worked like this but discuss that with Valeh definitely before it became too late
        super(EncodePatches, self).__init__()
        self.num_patches = num_patches
        self.project_dim = project_dim
        self.patch_dim = patch_dim
        self.project = torch.nn.Linear(patch_dim, project_dim)
        self.positioning = torch.nn.Embedding(num_patches, project_dim)
    
    def forward(self, patch):
        
        
        
        #positions = torch.range(start = 0, end = self.num_patches - 1, step = 1)
        positions = torch.arange(self.num_patches)
        #EncodedPatches = self.project(patch) + self.positioning(positions)
        positions = positions.to(device)
        patch = patch.to(device)
        EncodedPatches = torch.add(self.project(patch), self.positioning(positions))
        #EncodedPatches = self.project(patch)
        return EncodedPatches
    
    
class MultiLayerPerceptronForTransformerBlock(torch.nn.Module):
    def __init__(self, project_dim):
        super(MultiLayerPerceptronForTransformerBlock, self).__init__()
        self.project_dim = project_dim
        self.FirstHiddenLayer = torch.nn.Linear(project_dim, project_dim*2)
        self.SecondHiddenLayer = torch.nn.Linear(project_dim*2, project_dim)
        self.ActivationFunction = torch.nn.GELU()
        self.DropOut = torch.nn.Dropout(p = 0.1)
        
    def forward(self, x):
        
        x = self.FirstHiddenLayer(x)
        x = self.ActivationFunction(x)
        x = self.DropOut(x)
        x = self.SecondHiddenLayer(x)
        x = self.ActivationFunction(x)
        x = self.DropOut(x)
        
        return x
    
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, num_heads, project_dim, num_patches ):
        super(TransformerBlock, self).__init__()
        self.num_heads, self.project_dim, self.num_patches = num_heads, project_dim, num_patches
        self.Normalization_1 = torch.nn.LayerNorm(project_dim, eps=1e-06)
        self.Attention = torch.nn.MultiheadAttention(embed_dim = project_dim, num_heads = num_heads, dropout = 0.1, batch_first=True)
        self.TransfomerMLP = MultiLayerPerceptronForTransformerBlock(project_dim)
        self.Normalization_2 = torch.nn.LayerNorm(project_dim, eps=1e-06)
            
        
    def forward(self, encoded_patches):
        x1 = self.Normalization_1(encoded_patches)
        attention_out, attention_weights = self.Attention(x1, x1, x1, average_attn_weights=True)
        x2 = torch.add(attention_out, encoded_patches)
        x3 = self.Normalization_2(x2)
        x3 = self.TransfomerMLP(x3)
        
        encoded_patches = torch.add(x3, x2)
        
        return encoded_patches
    
    
class MultiLayerPerceptronForHead(torch.nn.Module):
    def __init__(self, num_patches, project_dim):
        super(MultiLayerPerceptronForHead, self).__init__()
        self.num_patches, self.project_dim = num_patches, project_dim
        self.FirstHiddenLayer = torch.nn.Linear(num_patches*project_dim, 2048)
        self.SecondHiddenLayer = torch.nn.Linear(2048, 1024)
        self.ThirdHiddenLayer = torch.nn.Linear(1024, 512)
        self.ForthHiddenLayer = torch.nn.Linear(512, 64)
        self.FifthHiddenLayer = torch.nn.Linear(64, 32)
        self.ActivationFunction = torch.nn.GELU()
        self.DropOut = torch.nn.Dropout(p = 0.3)
        
    def forward(self, x):
        
        x = self.FirstHiddenLayer(x)
        x = self.ActivationFunction(x)
        x = self.DropOut(x)
        x = self.SecondHiddenLayer(x)
        x = self.ActivationFunction(x)
        x = self.DropOut(x)
        x = self.ThirdHiddenLayer(x)
        x = self.ActivationFunction(x)
        x = self.DropOut(x)
        x = self.ForthHiddenLayer(x)
        x = self.ActivationFunction(x)
        x = self.DropOut(x)
        x = self.FifthHiddenLayer(x)
        x = self.ActivationFunction(x)
        x = self.DropOut(x)
        
        return x
    

class VitModel(torch.nn.Module):
    def __init__(self, patch_size, num_patches, project_dim, num_heads):
        super(VitModel, self).__init__()
        
        self.PacthesCreate = PatchesCreate(patch_size)
        self.EncodePatches = EncodePatches(num_patches, project_dim, patch_size**2*3)
        self.TB_1 = TransformerBlock(num_heads, project_dim, num_patches)
        self.TB_2 = TransformerBlock(num_heads, project_dim, num_patches)
        self.TB_3 = TransformerBlock(num_heads, project_dim, num_patches)
        self.TB_4 = TransformerBlock(num_heads, project_dim, num_patches)
        self.Normalization = torch.nn.LayerNorm(project_dim, eps=1e-06)
        self.MPH = MultiLayerPerceptronForHead(num_patches, project_dim)
        self.DropOut = torch.nn.Dropout(p = 0.3)
        self.bounding_box = torch.nn.Linear(patch_size, 4)
        
    def forward(self, inputs):
        
        patches = self.PacthesCreate(inputs)
        
        encoded_patches = self.EncodePatches(patches)
        
        # Four transformer block (Can it be written in a better way ask to Valeh!!!)
        encoded_patches = self.TB_1(encoded_patches)
        encoded_patches = self.TB_2(encoded_patches)
        encoded_patches = self.TB_3(encoded_patches)
        encoded_patches = self.TB_4(encoded_patches)
        
        # Head
        representation = self.Normalization(encoded_patches)
        representation = torch.flatten(representation, start_dim = 1)
        representation = self.DropOut(representation)
        # MLP
        features = self.MPH(representation)
        
        # Neurons that output Bounding Box
        bounding_box = self.bounding_box(features)
        
        return bounding_box
    