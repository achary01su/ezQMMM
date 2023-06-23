######################################################################
## QM/MM set up for QChem without bond breaking
######################################################################
puts "-----------------------------------------------"
puts "The tcl script requires one input file."
puts "For example, vmd -dispdev text -e <>.tcl -args <input file>"
puts "Copyright 2023 Atanu Acharya"
puts "Contributors: Atanu Acharya"
puts "Version updated on June 22, 2023"
puts "-----------------------------------------------"


######################################################################
## Process the input file
######################################################################
set fileName [lindex $argv 0]
puts "Input filename: $fileName"
set fileN [open $fileName r]

######################################################################
## Parse the input file
######################################################################
while {[gets $fileN line] != -1} {
    # Skip lines starting with #
    if {[string match "#*" $line]} {
        continue
    }
    # Strip =
    set stripped_line [string map {= " "} $line]

    # Trim spaces
    set trim_line0 [string trim [lindex $stripped_line 0]]
    set trim_line1 [string trim [lindex $stripped_line 1]]

    # Generate variable name and assign value
    set $trim_line0 $trim_line1
}

## Done processing and reading the input file
close $fileN

## Open a LOGFILE for STDOUT of the script
set LOGFILE [open "${OutputPrefix}_QMMM_details.log" w+]

######################################################################
## Check if variable values are set or not
## Specify the variables to check
######################################################################
set variablesToCheck {
    {ToolScriptPath} {} 
    {OutputDir} {}
    {OutputPrefix} {} 
    {QM1} {}
    {Charge} {0} 
    {Multiplicity} {1}
    {QChemRemFile} {}
    {PSFfile} {}
    {NumDCDs} {1}
    {DCDfile1} {}
    {FirstFrame} {0}
    {LastFrame} {-1}
    {TimeStep} {}
    {STEPsize} {}
    {FrameFreq} {}
}


foreach {variable defaultValue} $variablesToCheck {
    if {![info exists $variable]} {
        if {[info exists defaultValue]} {
            puts $LOGFILE "WARNING: Variable '$variable' is not set."
            puts $LOGFILE "WARNING: Using '$variable = $defaultValue' as default"
            set $variable $defaultValue
        } else {
            puts "Error: Variable '$variable' is not set and no default value is specified."
	    puts "Set the vaiable name in the input file"
	    quit
        }
    }
}
######################################################################


######################################################################
## Print the extracted variables for QM/MM set up
######################################################################
puts $LOGFILE "---------------Variable Summary---------------------"
puts $LOGFILE "ToolScriptPath = $ToolScriptPath"
puts $LOGFILE "OutputDir = $OutputDir"
puts $LOGFILE "OutputPrefix = $OutputPrefix"
puts $LOGFILE " "
puts $LOGFILE "QM1 = {$QM1}"
puts $LOGFILE "Charge = $Charge"
puts $LOGFILE "Multiplicity = $Multiplicity"
puts $LOGFILE "QChemRemFile = $QChemRemFile"
puts $LOGFILE " "
puts $LOGFILE "PSFfile = $PSFfile"
puts $LOGFILE "NumDCDs = $NumDCDs"
if {$NumDCDs > 1 } {
	for {set i 1} {$i <= $NumDCDs} {incr i} {
		set tempDCD [set DCDfile$i]
		puts $LOGFILE "DCDfile${i} =  $tempDCD"
	}
} else {
	puts $LOGFILE "DCDfile1 =  $DCDfile1"
  }
puts $LOGFILE "FirstFrame = $FirstFrame"
puts $LOGFILE "LastFrame = $LastFrame"
puts $LOGFILE " "
puts $LOGFILE "TimeStep = $TimeStep fs"
puts $LOGFILE "STEPsize = $STEPsize"
puts $LOGFILE "FrameFreq = $FrameFreq"
puts $LOGFILE "----------------------------------------------------"
######################################################################

## Load the useful-tcl-codes from the lab github page
source $ToolScriptPath 

######################################################################
## Create the output directory
######################################################################
if {[file isdirectory $OutputDir] != 1} {
     file mkdir $OutputDir 
} else {
   puts $LOGFILE "WARNING: Requested directory already exists"
}
######################################################################

############### No hard coded variables after this #############

######################################################################
## Load the psffile and dcdfile
mol new $PSFfile
if {$NumDCDs > 1 } {
	for {set i 1} {$i <=$NumDCDs} {incr i} {
		set tempDCD [set DCDfile$i]
		mol addfile $tempDCD type dcd step $STEPsize first $FirstFrame last $LastFrame waitfor all
	}
} else {
	mol addfile $DCDfile1 type dcd step $STEPsize first $FirstFrame last $LastFrame waitfor all

  }
######################################################################

######################################################################
## unwrap and wrap the trajectory
######################################################################
package require pbctools
pbc unwrap -all

load /home/achary01/softwares/bin/qwrap/qwrap.so
package require qwrap
qwrap compound fragment center "$QM1"
######################################################################


######################################################################
## Allign the system.
## This is not necessary but it is a cleaner approach
######################################################################
set ref [atomselect top "$QM1" frame 0]
set comp [atomselect top "$QM1"]
set all [atomselect top all]

for {set frame 0} {$frame <= [molinfo top get numframes]} {incr frame} {
    $comp frame $frame
    $all frame $frame
    set trans_mat [measure fit $comp $ref]
    $all move $trans_mat
}
$ref delete
$comp delete
$all delete
######################################################################

######################################################################
## Details of the systems and simulations
######################################################################
set n_frames [molinfo top get numframes]
set n_qm_atoms [numatoms top "$QM1"]
set n_mm_atoms [numatoms top "not ($QM1)"]

puts $LOGFILE "Number of frames: $n_frames"
puts $LOGFILE "QM selection: ${QM1}"
puts $LOGFILE "Number of QM atoms: $n_qm_atoms"
puts $LOGFILE "Number of MM atoms: $n_mm_atoms"

set timestep [format {%.3f} [expr $n_frames  * $TimeStep * $STEPsize * $FrameFreq/ 1000.000]] ; # in ps
puts $LOGFILE "Simulation time: $n_frames frames =  $timestep ps" 
######################################################################

for {set i 0} {$i < $n_frames} {incr i} {
    set currstep [expr {${i} * $TimeStep * $STEPsize * $FrameFreq/1000.000}] ; # in ps
    set qcfile [open "${OutputDir}/${OutputPrefix}_${currstep}.inp" w]
    set QMAtoms [parseAtoms top $i "$QM1"]

    #  Slurp up the $rem text file file
    set fp [open "$QChemRemFile" r]
    set file_data [read $fp]
    close $fp

    #  Process data file
    set data [split $file_data "\n"]
    foreach line $data {
	    puts $qcfile "$line"
    }

    puts $qcfile "\$molecule"
    puts $qcfile "$Charge $Multiplicity"
    dict for {id info} $QMAtoms {
    	dict with info {
	   set atom [string index $name 0]
    	   puts $qcfile "$atom    $xcoor $ycoor $zcoor"
	}
    }
    puts $qcfile "\$end"
    puts $qcfile ""

    ## Now add the MM point charges
    set MMAtoms [parseAtoms top $i "not ($QM1)"]
    puts $qcfile "\$external_charges"

    dict for {id info} $MMAtoms {
        dict with info {
           puts $qcfile "$xcoor $ycoor $zcoor $charge"
        }
    }

    puts $qcfile "\$end"
    close $qcfile
    puts "Created QChem QM/MM input file: ${OutputDir}/${OutputPrefix}_${currstep}.inp for frame at ${currstep} ps"
}

close $LOGFILE
quit


