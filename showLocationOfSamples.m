function showLocationOfSamples( image, rowAndColNumbers, t )
%

%
colors = [ 'b', 'r' ];
markers = [ 'o', 'x' ];

%
%imagesc( image ), colormap gray         
imshow( image )
hold on
T = [ t 1-t ];
for k = 1 : 2
  tmp = rowAndColNumbers( find( T(:,k) ), : );
  s = scatter( tmp(:,2), tmp(:,1) );
  marker = markers( k );
  color = colors( k );
  set( s, 'Marker', marker, 'MarkerEdgeColor', color )
  %set( s, 'LineWidth', 1.5 )
  set( s, 'LineWidth', 0.75 )
end
%set( gca, 'XTickLabel', [], 'YTickLabel', [] )

