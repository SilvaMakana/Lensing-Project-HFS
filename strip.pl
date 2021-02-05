while ($line = <>) {
	$line =~ s/\[//;
	$line =~ s/\[//;
	$line =~ s/\]//;
	$line =~ s/\]//;
	print $line;
}
