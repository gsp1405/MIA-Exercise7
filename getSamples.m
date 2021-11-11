function [ x, t, rowAndColNumbers ] = getSamples( featureImage, segmentation, numberOfSamples, mask )
%


if nargin < 4
  mask = ones( size( segmentation ) );
end

%
numberOfFeatures = size( featureImage, 3 );
numberOfPixels = size( featureImage, 1 ) * size( featureImage, 2 );

numberOfSamplesPerClass = ceil( numberOfSamples / 2 );
x = zeros( numberOfSamplesPerClass, numberOfFeatures, 2 );
t = zeros( numberOfSamplesPerClass, 2 );
rowAndColNumbers = zeros( numberOfSamplesPerClass, 2, 2 );


% Let's try to be balanced: approximately the same number of samples from each class
oneHotEncoding = zeros( size( segmentation, 1 ), size( segmentation, 2 ), 2 );
oneHotEncoding( :, :, 1 ) = segmentation;
oneHotEncoding( :, :, 2 ) = 1-segmentation;
[ rows, cols ] = ndgrid( 1:size( featureImage, 1 ), 1:size( featureImage, 2 ) );
for k=1:2

  indices = find( oneHotEncoding( :, :, k ) .* mask );
  sampleIndices = randperm( length( indices ), numberOfSamplesPerClass );
  indices = indices( sampleIndices );

  tmp = reshape( featureImage, [ numberOfPixels numberOfFeatures ] );
  x( :, :, k ) = tmp( indices, : );
  t( :, k ) = segmentation( indices );
  rowAndColNumbers( :, :, k ) = [ rows( indices ) cols( indices ) ];
end

%
x = [ x( :, :, 1 ); x( :, :, 2 ) ];
t = t( : );
rowAndColNumbers = [ rowAndColNumbers( :, :, 1); rowAndColNumbers( :, :, 2); ]

%
x = x( 1 : numberOfSamples, : );
t = t( 1 : numberOfSamples, : );
rowAndColNumbers = rowAndColNumbers( 1 : numberOfSamples, : );

