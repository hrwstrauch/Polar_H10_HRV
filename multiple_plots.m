
FolderName = '/Users/hans_apple/data/test_claude_code/tempdir';   % Your destination folder
FigList = findobj(allchild(0), 'flat', 'Type', 'figure');
for iFig = 1:length(FigList)
  FigHandle = FigList(iFig);
  FigName   = num2str(get(FigHandle, 'Number'));
  set(0, 'CurrentFigure', FigHandle);
  saveas(FigHandle,fullfile(FolderName, [FigName '.fig'])); %Specify format for the figure
end