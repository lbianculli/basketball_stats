import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns


class PlotSeason():
    sns.set()
    
    def __init__(self, data): 
        self.data = data
        
    def scatter_plot(self, X_name, y_name, top_n=None, best_fit=True,
                          annotation=1, figsize=(16,6)):
        
        cmap = cm.get_cmap('plasma')

        plt.figure(figsize=figsize)

        left = bottom =.25
        width = .5
        height = 1

        axes_shape = [left, bottom, width, height]
        ax1 = plt.axes(axes_shape)

        data = self.data.sort_values('MP', ascending=False).head(top_n)
        X = np.asarray(data[X_name], dtype=np.float)
        y = np.asarray(data[y_name], dtype=np.float)

        cmap = cmap(minmax_scale(X)- .1)

        ax1.scatter(X, y, s=50, c=cmap)


        xticks = ax1.get_xticklabels() #get tick labels so we can adjust them a bit with .setp
        plt.setp(xticks, rotation=45)
        
        if best_fit is True:
            ax1.plot(np.unique(X), np.poly1d(np.polyfit(X, y, 1))(np.unique(X)))
            
        ax1.set(xlabel=X_name, ylabel=y_name, 
                title="{} vs. {}".format(X_name, y_name))

        if annotation is not None:
            if isinstance(annotation, str):
                try:
                    player_idx = data.index.get_loc(annotation)
                    ax1.annotate(annotation, (X[player_idx], y[player_idx]), xytext=(-10,-10),
                                textcoords='offset pixels', size=8) #change style later

                except Exception as e:
                    print('{} is not in the sample. Did you enter the name correctly?'.format(e))

            if isinstance(annotation, int):
                data_top = data.sort_values(X_name, ascending=False).head(annotation)
                data_bottom = data.sort_values(X_name, ascending=True).head(annotation)
                names_top = [i for i in data_top.index][::-1]
                names_bottom = [i for i in data_bottom.index] #do i need to reverse this?
                array = np.array(data[X_name])
                X_top = (array.argsort()[-annotation:]) #do i need the astype?
                X_bottom = (array.argsort()[:annotation])
                new_names_top = []
                new_names_bottom = []

                for i in names_top:
                    name = i.split(' ')
                    new_names_top.append(name[0][0] + '. ' + name[1])

                for i in names_bottom:
                    name = i.split(' ')
                    new_names_bottom.append(name[0][0] + '. ' + name[1])

                array = np.array(data[X_name])
                array_sort = (array.argsort()[-3:]).astype(int)

                for i, x_idx in enumerate(X_top):
                    ax1.annotate(new_names_top[i], xy=(X[x_idx], y[x_idx]), xytext=(-10,-10),
                            arrowprops=dict(arrowstyle='->'), textcoords='offset pixels', size=8) 

                for i, x_idx in enumerate(X_bottom):
                    ax1.annotate(new_names_bottom[i], xy=(X[x_idx], y[x_idx]), xytext=(-10,-10),
                            arrowprops=dict(arrowstyle='->'), textcoords='offset pixels', size=8) 
                    
                    
    def joint_plot(self, X_name, y_name, best_fit=True, top_n=150,
                   kde=True, figsize=(15,5)): 
        

        cmap = cm.get_cmap('plasma')

        plt.figure(figsize=figsize)

        left = bottom =.25
        width = .5
        height = 1
        left_hist = width + .25
        bottom_hist = height + .25

        axes_shape = [left, bottom, width, height]
        histx_shape = [left, bottom_hist, width, .11]
        histy_shape = [left_hist, bottom, .075, height]
        ax1 = plt.axes(axes_shape)
        ax_histx = plt.axes(histx_shape)
        ax_histy = plt.axes(histy_shape)

        data = self.data.sort_values('MP', ascending=False).head(top_n)
        X = np.asarray(data[X_name], dtype=np.float)
        y = np.asarray(data[y_name], dtype=np.float)

        cmap = cmap(minmax_scale(X)- .1)

        ax1.scatter(X, y, s=50, c=cmap)
        ax_histx.hist(X, bins=25, color='grey')
        ax_histx.axis('off')
        ax_histy.hist(y, bins=25, orientation='horizontal', color='grey')
        ax_histy.axis('off')



        xticks = ax1.get_xticklabels()
        plt.setp(xticks, rotation=45)

        if best_fit is True:
            ax1.plot(np.unique(X), np.poly1d(np.polyfit(X, y, 1))(np.unique(X)))

        ax1.set(xlabel=X_name, ylabel=y_name,);
        
    
    def multifacet(self, X_name, y_name, col=None, hue=None, top_n=None): 
        """
        Creates multiple scatter plots based on [categorical] labels col_name and hue. Either can be specified.
        """
        
        data = self.data.sort_values('MP', ascending=False).head(top_n)
        data[X_name] = data[X_name].astype(float)
        data[y_name] = data[y_name].astype(float)
        X = np.asarray(data[X_name])
        y = np.asarray(data[y_name])

        if col is None: 
            col_wrap = None
        else:
            col_wrap = 2

        facet = sns.FacetGrid(data, col=col, hue=hue, col_wrap=col_wrap, height=4.5, aspect=1.25, palette='plasma')
        facet.map(sns.regplot, X_name, y_name, ci=None, color='purple').add_legend()
