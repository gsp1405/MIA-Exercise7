function featureImage = getInputFeatures( image, numberOfNeighbors, shift )
%

numberOfFeatures = numberOfNeighbors + 1;

if nargin < 3
  shift = 1;
end


% 
leftNeighborImage = zeros( size( image ) );
leftNeighborImage( :, shift+1:end ) = image( :, 1:end-shift );

rightNeighborImage = zeros( size( image ) );
rightNeighborImage( :, 1:end-shift ) = image( :, shift+1:end );

topNeighborImage = zeros( size( image ) );
topNeighborImage( shift+1:end, : ) = image( 1:end-shift, : );

bottomNeighborImage = zeros( size( image ) );
bottomNeighborImage( 1:end-shift, : ) = image( shift+1:end, : );

leftTopNeighborImage = zeros( size( image ) );
leftTopNeighborImage( shift+1:end, shift+1:end ) = image( 1:end-shift, 1:end-shift );

rightTopNeighborImage = zeros( size( image ) );
rightTopNeighborImage( shift+1:end, 1:end-shift ) = image( 1:end-shift, shift+1:end );

leftBottomNeighborImage = zeros( size( image ) );
leftBottomNeighborImage( 1:end-shift, shift+1:end ) = image( shift+1:end, 1:end-shift );

rightBottomNeighborImage = zeros( size( image ) );
rightBottomNeighborImage( 1:end-shift, 1:end-shift ) = image( shift+1:end, shift+1:end );




if numberOfNeighbors == 0
  % One feature per voxel only: intensity of voxel itself
  features = [ image( : ) ];
  
elseif numberOfNeighbors == 1
  % First feature is intensity of voxel itself, second one is intensity of bottom neighbor
  features = [ image( : ) bottomNeighborImage( : ) ];
  
elseif numberOfNeighbors == 4
  %
  features = [ image( : ), ...
               leftNeighborImage( : ), rightNeighborImage( : ), ...
               topNeighborImage( : ), bottomNeighborImage( : )  ...
             ];
  
elseif numberOfNeighbors == 8
  %
  features = [ image( : ), ...
               leftTopNeighborImage( : ), topNeighborImage( : ), rightTopNeighborImage( : ), ...
               leftNeighborImage( : ), rightNeighborImage( : ), ...
               leftBottomNeighborImage( : ), bottomNeighborImage( : ), rightBottomNeighborImage( : )  ...
             ];

else
  %
  error( 'Not implemented yet' )
  
end


featureImage = reshape( features, [ size( image, 1 ), size( image, 2 ),  numberOfFeatures ] );

