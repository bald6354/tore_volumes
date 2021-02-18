classdef reshapeLayer < nnet.layer.Layer
    
    properties
        % (Optional) Layer properties.
        
        % Layer properties go here.
        inSize
        outSize
    end
    
    properties (Learnable)
        % (Optional) Layer learnable parameters.
        
        % Layer learnable parameters go here.
    end
    
    methods
        function layer = reshapeLayer(name, inSize, outSize)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            % Layer constructor function goes here.
            
            %In the future make this an input variable
%             numMetadata = 3;
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Reshape Layer";
            
            % Initialize scaling coefficient.
            layer.inSize = inSize;
            layer.outSize = outSize;
            
            % Capture input, all but last layer, and convert last layer
            % into a vector of length numMetadata
%             layer.NumOutputs = 1 + 1;
            
%             layer.OutputNames = {'imageOut', 'metaOut'};
            %             % Dissociate layer doesn't need X or Z for the backward pass
            %             layer.NeedsXForBackward = false;
            %             layer.NeedsZForBackward = false;
        end
        
        function [Z1] = predict(layer, X1)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            
            %Dissociate the last layer (which contains only metadata)
            Z1 = reshape(X1, layer.outSize(1), layer.outSize(2), layer.outSize(3), layer.outSize(4), []);

        end
        
        function [dX] = backward(layer, ~, ~, dZ1, ~)
            
            dX = reshape(dZ1, layer.inSize(1), layer.inSize(2), layer.inSize(3), layer.inSize(4), []);
            
            % There are no learnable parameters.
            %             dW = [];
        end
    end
end