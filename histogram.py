import numpy as np
from sklearn.linear_model import LinearRegression

import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# The estimator class takes the bin frequencies and uses a splines
# approximation algorithm to create a curve that best fits the bins
# frequencies.
class Estimator:
    def fit( this, x, y, kn ):
        A = [ [] for _ in range( len( x ) ) ]

        xmin = min( x )
        xmax = max( x )

        step = ( xmax - xmin ) / ( kn + 1 )
        this.kn = kn
        this.k = [ xmin + step ]
        for ki in range( kn - 1 ):
            this.k.append( this.k[ ki ] + step )

        xi = 0
        ki = 0

        while( xi < len( x ) ):
            if( ki < kn and x[ xi ] >= this.k[ ki ] ):
                ki += 1
            else:
                for i in range( 4 ):
                    A[ xi ].append( x[ xi ] ** i )
                for i in range( 4, 4 + ki ):
                    A[ xi ].append( ( x[ xi ] - this.k[ i - 4 ] ) ** 3 )
                for i in range( 4 + ki, 4 + kn ):
                    A[ xi ].append( 0 )
                xi += 1
                
        model = LinearRegression( fit_intercept = False ).fit( A, y )
        this.q = model.coef_

    def produce( this, xmin, xmax, p ):
        x = np.linspace( xmin, xmax, p )
        xy = []

        xi = 0
        ki = 0

        while( xi < x.shape[ 0 ] ):
            if( ki < this.kn and x[ xi ] >= this.k[ ki ] ):
                ki += 1
            else:
                yi = 0
                for i in range( 4 ):
                    yi += this.q[ i ] * ( x[ xi ] ** i )
                for i in range( 4, 4 + ki ):
                    yi += this.q[ i ] * ( ( x[ xi ] - this.k[ i - 4 ] ) ** 3 )
                xy.append( x[ xi ] )
                xy.append( yi )
                xi += 1

        return xy

# This class defines some methods to calculate weighted statistics
class WeightedStats:
    # Values and weights are initialized
    def __init__( this, values, weights ):
        this.values = values
        this.weights = weights
        this.stats_vector = [ 0 for _ in range( 3 ) ] # There are three different statistics (mean, std, pm)
    
    # Mean
    def mean( this ):
        m = 0.0
        for i in range( len( this.values ) ):
            m += this.values[ i ] * this.weights[ i ]
        return m / sum( this.weights )
    
    # Standard deviation
    def std( this ):
        m = this.mean()
        s = 0.0
        for i in range( len( this.values ) ):
            s += this.weights[ i ] * ( abs( this.values[ i ] - m ) ** 2 )
        return ( s / sum( this.weights ) ) ** ( 1 / 2 )

    # Polarization measure (as proposed by Esteban & Ray)
    def pm( this, K = 1, a = 1.6 ):
        p = 0.0
        for i in range( len( this.values ) ):
            for j in range( len( this.values ) ):
                p += ( ( this.weights[ i ] ** ( 1 + a ) ) * this.weights[ j ] ) * abs( this.values[ i ] - this.values[ j ] )
        return K * p
    
    # Generates a vector that contains the statistics for a given histogram configuration
    def gen_stats_vector( this ):
        this.stats_vector[ 0 ] = this.mean()
        this.stats_vector[ 1 ] = this.std()
        this.stats_vector[ 2 ] = this.pm()

# This class defines the histogram. It deals with it's visualization and behavior
class Histogram:
    # Determines the index of a bin on the bin array based on its index
    def where_id_is( this, id ):
        for i in range( this.bins ):
            if( this.bin_ids[ i ] == id ):
                return i
        return None

    # When left click is pressed, determine if user's mouse is over a bin, and mark it as selected
    def on_press( this, event ):
        this.selected_bin = this.canvas.find_withtag( "current" ) # Get element under mouse
        if( len( this.selected_bin ) > 0 and this.selected_bin[ 0 ] != this.histogram_frame ):
            this.selected_bin = this.selected_bin[ 0 ] # Mark bin as selected
            this.bin_on_click = this.canvas.coords( this.selected_bin ) # Store bin position
            this.mouse_on_click = ( event.x, event.y ) # Store mouse position on click
        else:
            this.selected_bin = None # No bin was selected
    
    # When mouse moves (while click on hold), update bin data and display based on distance traveled by mouse
    def on_move_press( this, event ):
        bin_index = this.where_id_is( this.selected_bin ) # Get bin index on array based on its id (to update frequency)
        if( bin_index != None and
            this.bin_on_click[ 1 ] - ( this.mouse_on_click[ 1 ] - event.y ) < this.canvas_height - this.hmargin_to_plot and
            this.bin_on_click[ 1 ] - ( this.mouse_on_click[ 1 ] - event.y ) > this.hmargin_to_plot
        ):
            # Distance traveled by the mouse is calculated and histogram is updated accordingly
            dif = this.bin_on_click[ 3 ] - ( this.bin_on_click[ 1 ] - ( this.mouse_on_click[ 1 ] - event.y ) )
            n = round( dif / this.height_unit )
            this.canvas.coords( this.selected_bin, this.bin_on_click[ 0 ], this.bin_on_click[ 3 ] - n * this.height_unit, *this.bin_on_click[ 2 : 4 ] )
            this.px_y_vals[ bin_index ] = this.bin_on_click[ 3 ] - n * this.height_unit
            this.draw_graph()
            this.active_weights[ bin_index ] = n
            this.update_stats()

    # Every time a change is triggered in the histogram, statistics must be updated
    def update_stats( this ):
        stats = WeightedStats( this.active_values, this.active_weights ) # Set values and weights
        if( sum( this.active_weights ) > 0 ): # Sum of weights must be greater than zero for mean and std calculation
            mean = stats.mean()
            std = stats.std()
            this.mean_display.config( text = f"MEAN = {mean:.1f}" ) # Updates GUI
            this.std_display.config( text = f"STD = {std:.1f}" ) # Updates GUI
        else: # Set statistics to undefined
            this.mean_display.config( text = f"MEAN = UNDEFINED" ) # Updates GUI
            this.std_display.config( text = f"STD = UNDEFINED" ) # Updates GUI
        pm = stats.pm()
        this.pm_display.config( text = f"PM = {pm:.1f}" ) # Updates GUI

    # Initializes histogram. There is a bunch code here that just makes sure the histogram looks good
    def init_histogram( this ):
        ############################################################################################
        # Creates the canvas that holds the histogram, and the histogram viewport
        ############################################################################################
        this.yspace = this.canvas_height - 2 * this.hmargin_to_plot
        this.xspace = this.canvas_width - 2 * this.wmargin_to_plot
        this.height_unit = this.yspace / this.max_freq
        this.bin_width = this.xspace / this.bins
        this.histogram_frame = this.canvas.create_rectangle( this.wmargin_to_frame,
                                                        this.hmargin_to_frame,
                                                        this.canvas_width - this.wmargin_to_frame,
                                                        this.canvas_height - this.hmargin_to_frame,
                                                        fill = "white" )
        ############################################################################################
        # Creates the scales for the histogram (for x and y axes)
        ############################################################################################
        this.yaxis_text_ids = []
        this.yaxis_line_ids = []
        this.px_x_vals = [] # For estimator
        this.px_y_vals = [] # For estimator
        this.graph_line_ids = [] # For estimator
        hbase = this.hmargin_to_frame + this.canvas_height * 0.05
        for i in range( this.max_freq + 1 ):
            this.yaxis_text_ids.append(
                this.canvas.create_text(
                    ( this.wmargin_to_frame - this.bin_width * 0.8, hbase + this.height_unit * i ),
                    text = f"{ this.max_freq - i }"
                )
            )
            this.yaxis_line_ids.append(
                this.canvas.create_line(
                    this.wmargin_to_frame - this.bin_width * 0.1,
                    hbase + this.height_unit * i,
                    this.wmargin_to_frame,
                    hbase + this.height_unit * i
                )
            )
        ############################################################################################
        # Creates each bin (rectangle) and also the bin array (for frequencies) and populates it
        ############################################################################################
        this.bin_ids = []
        this.standby_values = [ 0 for _ in range( this.bins ) ]
        this.standby_weights = [ 0 for _ in range( this.bins ) ]
        this.active_values = [ 0 for _ in range( this.bins ) ]
        this.active_weights = [ 0 for _ in range( this.bins ) ]
        for i in range( this.bins ):
            this.bin_ids.append( this.canvas.create_rectangle( this.wmargin_to_plot + i * this.xspace / this.bins,
                                          this.canvas_height - this.hmargin_to_plot - this.height_unit,
                                          this.wmargin_to_plot + ( i + 1 ) * this.xspace / this.bins,
                                          this.canvas_height - this.hmargin_to_plot,
                                          fill = "blue" ) )
            this.active_weights[ i ] = 1.0
            this.active_values[ i ] = 1.0 / ( this.bins * 2 ) + i / this.bins
        ############################################################################################
            this.px_x_vals.append( this.wmargin_to_plot + i * this.xspace / this.bins + this.bin_width / 2 )
            this.px_y_vals.append( this.canvas_height - this.hmargin_to_plot - this.height_unit )
        this.draw_graph()

    # Constructor method. Allows for histogram to have a parametric amount of bins and max frequency of each bin
    def __init__( this, app, width, height, bins = 10, max_freq = 10 ):
        this.kn = 10
        this.mean_display = app.mean_display # Needed to display mean
        this.std_display = app.std_display # Needed to display std
        this.pm_display = app.pm_display # Needed to display pm

        #########################################################
        # Variables for histogram setup (visual)
        #########################################################
        this.canvas_width = width
        this.canvas_height = height
        this.wmargin_to_frame = this.canvas_width * 0.1
        this.hmargin_to_frame = this.canvas_height * 0.1
        this.wmargin_to_plot = this.canvas_width * 0.15
        this.hmargin_to_plot = this.canvas_height * 0.15
        #########################################################

        this.bins = bins # Number of bins
        this.max_freq = max_freq # Max bin frequency

        # Histogram canvas is created
        this.canvas = tk.Canvas( app.main,
                                 width = this.canvas_width,
                                 height = this.canvas_height )
        
        # Binds mouse events to corresponding behavior
        this.canvas.bind( "<Button-1>", this.on_press )
        this.canvas.bind( "<B1-Motion>", this.on_move_press )
        
        this.init_histogram() # Initializes histogram
        this.update_stats() # Updates statistics based on initialization values
    
    # Resets histogram to initialization values (used after animation is run)
    def reset( this, width, height, bins = 10, max_freq = 10 ):
        this.canvas_width = width
        this.canvas_height = height
        this.wmargin_to_frame = this.canvas_width * 0.1
        this.hmargin_to_frame = this.canvas_height * 0.1
        this.wmargin_to_plot = this.canvas_width * 0.15
        this.hmargin_to_plot = this.canvas_height * 0.15
        this.bins = bins
        this.max_freq = max_freq
        this.canvas.delete( "all" )
        this.canvas.config( width = this.canvas_width, height = this.canvas_height )
        this.init_histogram()
    
    # Sets weights of histogram (used after animation is run)
    def set_weights( this, weights ):
        for i in range( this.bins ):
            this.active_weights[ i ] = weights[ i ]
            #bin_index = this.where_id_is( this.bin_ids[ i ] )
            bin_coords = this.canvas.coords( this.bin_ids[ i ] )
            this.canvas.coords(
                this.bin_ids[ i ],
                bin_coords[ 0 ],
                bin_coords[ 3 ] - weights[ i ] * this.height_unit,
                *bin_coords[ 2 : 4 ]
            )

    # Uses the Estimator class to create apoxximated curve and draws it
    def draw_graph( this ):
        for i in range( len( this.graph_line_ids ) - 1, -1, -1 ):
            this.canvas.delete( this.graph_line_ids[ i ] )
            this.graph_line_ids.pop()
        est = Estimator()
        est.fit( this.px_x_vals, this.px_y_vals, this.kn )
        xy = est.produce( this.px_x_vals[ 0 ], this.px_x_vals[ -1 ], 100 )
        this.graph_line_ids.append( this.canvas.create_line( *xy, width = 3, fill = "orange" ) )

# App class
class App:
    # Defines transitions for Esteban & Ray axioms animations
    # Each animation is modelled here as an array of bin frequencies
    # Each array is a 'snapshot' of the whole animation
    def define_animations( this ):
        this.animations = []
        this.animations.append(
            [ [ 10, 0, 0, 0, 5, 0, 0, 0, 5, 0 ],
              [ 10, 0, 0, 0, 4, 1, 0, 1, 4, 0 ],
              [ 10, 0, 0, 0, 3, 1, 2, 1, 3, 0 ],
              [ 10, 0, 0, 0, 2, 1, 4, 1, 2, 0 ],
              [ 10, 0, 0, 0, 1, 1, 6, 1, 1, 0 ],
              [ 10, 0, 0, 0, 0, 1, 8, 1, 0, 0 ],
              [ 10, 0, 0, 0, 0, 0, 10, 0, 0, 0 ] ]
        )
        this.animations.append(
            [ [ 10, 0, 0, 0, 0, 3, 0, 0, 0, 6 ],
              [ 10, 0, 0, 0, 0, 2, 1, 0, 0, 6 ],
              [ 10, 0, 0, 0, 0, 1, 1, 1, 0, 6 ],
              [ 10, 0, 0, 0, 0, 0, 1, 2, 0, 6 ],
              [ 10, 0, 0, 0, 0, 0, 0, 3, 0, 6 ] ]
        )
        this.animations.append(
            [ [ 6, 0, 0, 0, 0, 8, 0, 0, 0, 6 ],
              [ 6, 0, 0, 0, 1, 6, 1, 0, 0, 6 ],
              [ 6, 0, 0, 1, 1, 4, 1, 1, 0, 6 ],
              [ 6, 0, 1, 1, 1, 2, 1, 1, 1, 6 ],
              [ 6, 1, 1, 1, 1, 0, 1, 1, 1, 7 ],
              [ 7, 1, 1, 1, 0, 0, 0, 1, 1, 8 ],
              [ 8, 1, 1, 0, 0, 0, 0, 0, 1, 9 ],
              [ 9, 1, 0, 0, 0, 0, 0, 0, 0, 10 ],
              [ 10, 0, 0, 0, 0, 0, 0, 0, 0, 10 ], ]
        )

    # Updates histogram based on selected 'snapshot' of a given animation
    def animation_routine( this, animation, select ):
        this.histogram.set_weights( animation[ select ] ) # Sets weights of histogram based on 'snapshot'
        this.histogram.update_stats() # Updates statistics
        stats = WeightedStats( this.histogram.active_values, this.histogram.active_weights )
        stats.gen_stats_vector() # Generate stat vector for statistics visualization (with matplotlib)
        for i in range( len( stats.stats_vector ) ):
            this.stats_progressions[ i ][ select ] = stats.stats_vector[ i ] # Adds metrics to stats progression (array that holds change of statistics over time)
        if( select < len( animation ) - 1 ):
            this.main.after( 1500, this.animation_routine, animation, select + 1 ) # Wait for 1.5 seconds before going again
        else:
            this.animation_running = False # Animation is over

    # Sets up everything for animation to run correctly
    def prepare_animation( this ):
        if( not this.animation_running ):
            this.animation_running = True
            this.histogram.reset( this.histogram.canvas_width, this.histogram.canvas_width )
            animation = this.animations[ this.animation_select.current() ]
            this.last_animation_size = len( animation )
            this.histogram.set_weights( animation[ 0 ] )
            this.histogram.update_stats()
            stats = WeightedStats( this.histogram.active_values, this.histogram.active_weights )
            stats.gen_stats_vector()
            for i in range( len( stats.stats_vector ) ):
                this.stats_progressions[ i ][ 0 ] = stats.stats_vector[ i ]
            this.main.after( 1500, this.animation_routine, animation, 1 )

    # Uses matplotlib to plot the way a statistic evolved over time during the animation of an axiom
    def plot_progressions( this ):
        if( not this.animation_running ): # Check that animation is not running
            progs = this.stats_progressions[ this.progression_select.current() ]
            fig = Figure( figsize = ( 5, 5 ), dpi = 50 )
            progs_plot = fig.add_subplot( 111 )
            progs_plot.plot( progs[ : this.last_animation_size ] )

            canvas = FigureCanvasTkAgg( fig, master = this.main )
            canvas.draw()

            canvas.get_tk_widget().place( x = 505, y = 500 * 0.1 + 150 )

    # Constructor method for the app
    def __init__( this ):
        this.define_animations() # Add animations
        this.stats_progressions = [ [ 0 for _ in range( 10 ) ] for _ in range( 3 ) ] # Creates progression array
        
        this.last_animation_size = 0 # Animation size is used to know when animation ends
        this.animation_running = False # Know if animation is running
        
        # Creates app window
        this.main = tk.Tk()
        this.main.geometry( "800x500" )
        
        # Creates displays for statistics
        this.mean_display = tk.Label( this.main, background = "white", borderwidth = 1, relief = "solid" )
        this.std_display = tk.Label( this.main, background = "white", borderwidth = 1, relief = "solid" )
        this.pm_display = tk.Label( this.main, background = "white", borderwidth = 1, relief = "solid" )

        # Creates histogram
        this.histogram = Histogram( this, 500, 500 )
        
        # Creates button to trigger animation
        this.animation_run = tk.Button( this.main, text = "Animate: ", command = this.prepare_animation )
        
        # Creates animation selector
        this.animation_select = ttk.Combobox( this.main, width = 8 )
        this.animation_select[ "values" ] = ( "Axiom-1", "Axiom-2", "Axiom-3" )
        this.animation_select[ "state" ] = "readonly"
        this.animation_select.current( 0 )
        
        # Creates gadget to select statistic progression that wants to be visualized
        this.view_progressions = tk.Button( this.main, text = "Plot: ", command = this.plot_progressions )
        this.progression_select = ttk.Combobox( this.main, width = 10 )
        this.progression_select[ "values" ] = ( "MEAN", "STD", "PM" )
        this.progression_select[ "state" ] = "readonly"
        this.progression_select.current( 0 )

        # Create selector for kn parameter of the estimator class
        this.kn_select = tk.Entry( this.main, text = "10", width = 10 )
        this.kn_select.place( x = 120, y = 15 )
        this.kn_change = tk.Button( this.main, text = "Use kn = ", command = this.change_kn )
        this.kn_change.place( x = 50, y = 15 )

        # Create layout of app
        this.histogram.canvas.place( x = 0, y = 0 )
        this.mean_display.place( x = 505, y = 500 * 0.1 )
        this.std_display.place( x = 505, y = 500 * 0.1 + 25 )
        this.pm_display.place( x = 505, y = 500 * 0.1 + 50 )
        this.animation_run.place( x = 500 * 0.1, y = 460 )
        this.animation_select.place( x = 500 * 0.1 + 70, y = 460 )
        this.view_progressions.place( x = 505, y = 500 * 0.1 + 100 )
        this.progression_select.place( x = 505 + 45, y = 500 * 0.1 + 100 )
    
    # Change value of the kn parameter of the estimator
    def change_kn( this ):
        try:
            kn = int( this.kn_select.get() )
        except:
            kn = 10
        finally:
            this.histogram.kn = kn
            this.histogram.draw_graph()

# Launch app
app = App()
app.main.mainloop()