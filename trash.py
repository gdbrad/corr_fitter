 
# from tabulate import tabulate

# abbreviations = [f[:-3] for f in os.listdir(input_dir) if f.endswith('.py')]

# for abbr in abbreviations:
#     fit_params = os.path.join(input_dir, f"{abbr}.py")
#     if not os.path.exists(fit_params):
#         print(f"Error: input file {fit_params} does not exist!")
#         continue

#     with open(fit_params, 'r') as f:
#         input_file_contents = f.read()

#     if 'p_dict' not in input_file_contents:
#         print(f"Error: input file {fit_params} does not contain a dictionary called 'p_dict'!")
#         continue

#     try:
#         p_dict = {}
#         exec(input_file_contents, p_dict)
#     except Exception as e:
#         print(f"Error: Failed to execute the contents of input file {fit_params}!\n{str(e)}")
#         continue

#     if 'tag' not in p_dict:
#         print(f"Warning: input file {fit_params} does not contain a dictionary called 'tag' within the 'p_dict' dictionary! Adding default values...")
#         p_dict['tag'] = {
#             'sigma' : 'sigma',
#             'sigma_st' : 'sigma_st',
#             'xi' :  'xi',
#             'xi_st' : 'xi_st',
#             'lam' : 'lam',
#         }

#     try:
#         hyperon_fit = fa.analyze_hyperon_corrs(data_file, fit_params, model_type=model_type,
#                                                bs=False, bs_file=bs_data_file,
#                                                bs_path=abbr, bs_N=bs_N, bs_seed=bs_seed)
#     except KeyError:
#         print(f"KeyError: Error analyzing hyperons for input file {fit_params}. Skipping abbreviation {abbr}.")
#         continue

#     my_fit = hyperon_fit.get_fit()

#     out_path = os.path.join(fit_results_dir, abbr, model_type)
#     ld.pickle_out(fit_out=my_fit, out_path=out_path, species="hyperons")
#     plot1 = hyperon_fit.return_best_fit_info()
#     plot2 = hyperon_fit.plot_effective_mass(t_plot_min=t_plot_min, t_plot_max=t_plot_max, model_type=model_type,
#                                             show_plot=True, show_fit=True)

#     output_dir = os.path.join(fit_results_dir, abbr, f"{model_type}_{abbr}")
#     os.makedirs(output_dir, exist_ok=True)
#     output_pdf = os.path.join(output_dir, 'output.pdf')
#     with PdfPages(output_pdf) as pp:
#         pp.savefig(plot1)
#         pp.savefig(plot2)

#     params_df = pd.DataFrame(my_fit.p).transpose()

#     print("Abbreviation:", abbr)
#     print(tabulate(params_df, headers='keys', tablefmt='fancy_grid'))

#     # Ask the user if the fit result is acceptable
#     while True:
#         result = input(f"Is the fit result for abbreviation {abbr} acceptable? (y/n): ")
#         if result == "y":
#             # Save the result and skip this fit in future runs
#             out_path = 'fit_results/{0}/{1}/'.format(abbr, model_type)
#             ld.pickle_out(fit_out=my_fit, out_path=out_path, species="hyperons")
#             break
#         elif result == "n":
#             break





# def plot_stability(self,pdf_filename,t_start=None,t_end=None,model_type=None,best_fit_t_idx=None,n_states_array=None):
    #     with PdfPages(pdf_filename) as pdf:
    #         colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    #         markers = ['o', 's', '^', 'v', 'D', 'P', 'X']
    #         cmap = matplotlib.cm.get_cmap('rainbow_r')
    #         norm = matplotlib.colors.Normalize(vmin=0.25, vmax=1.75)
    #         fit_data = {}
    #         t_middle = int((t_start + 2*t_end)/3)
    #         t = np.arange(t_start, t_middle + 1)
    #         for n_state in n_states_array:
    #             fit_data[n_state] = None
    #             for key in list(fit_data.keys()):
    #                 fit_data[key] = {}
    #                 for particle in self.p_dict['tag'].keys():
    #                     fit_data[key][particle] = {
    #                     'y' : np.array([]),
    #                     'chi2/df' : np.array([]),
    #                     'Q' : np.array([]),
    #                     't' : np.array([])
    #             }
    #         for n_state in list(fit_data.keys()):
    #             n_states_dict = self.n_states.copy()
    #             n_states_dict[model_type] = n_state
    #             for ti in t:
    #                 t_range = {key : self.t_range[key] for key in list(self.t_range.keys())}
    #                 t_range[model_type] = [ti, t_end]
    #                 print(t_range[model_type])
    #                 temp_fit = self.get_fit(t_range, n_states_dict)
    #                 print(temp_fit,"temp")
    #                 if temp_fit is not None:
    #                     if model_type == 'hyperons':
    #                         for particle in self.p_dict['tag'].keys():
    #                             fit_data[n_state][particle]['y'] = np.append(fit_data[n_state][particle]['y'], temp_fit.p[particle+'_E0'])
    #                             # fit_data[n_state]['chi2/df'] = np.append(fit_data[n_state]['chi2/df'], temp_fit.chi2 / temp_fit.dof)
    #                             # fit_data[n_state]['Q'] = np.append(fit_data[n_state]['Q'], temp_fit.Q)
    #                             fit_data[n_state][particle]['t'] = np.append(fit_data[n_state][particle]['t'], ti)
    #                             print(f"Appending data for particle {particle}, n_state {n_state}, time slice {ti}")
    #                             print("temp_fit.p[corr+'_E0']:", temp_fit.p[particle+'_E0'])
    #                             print("fit_data[n_state]['y']:", fit_data[n_state][particle]['y'])

            # spacing = (np.arange(len(n_states_array)) - (len(n_states_array)-1)/2.0)/((len(n_states_array)-1)/2.0) *0.25
            # for particle_idx, particle in enumerate(self.p_dict['tag'].keys()):
            #     fit_data[n_sta[particle] = {}
            #     for n_state in n_states_array:
            #         fit_data[particle][n_state] = {
            #         't' : np.array([]),
            #         'E0' : np.array([]),
            #         'E0_err' : np.array([]),
            #         'chi2/df' : np.array([]),
            #         'Q' : np.array([]),
            #         }
            #     # plt.figure(figsize=(10,10))
                

            #     plt.suptitle("Stability plot for %s, varying start\n Fitting [%s, %s], $N_{states} =$ %s"
            #         %(particle,"$t$", t_end, sorted(fit_data[particle].keys())), fontsize = 24)

            #     for n_state in list(fit_data[particle].keys()):
            #         n_states_dict = self.n_states.copy()
            #         n_states_dict[model_type] = n_state
            #         print(f"Checking n_state {n_state} at time slice {t}")  # New print statement
            #         for ti in t:
            #             t_range = {key : self.t_range[key] for key in list(self.t_range.keys())}
            #             t_range[model_type] = [ti, t_end]
            #             # fit_result = self.get_fit_forstab(t_range,n_states_dict)
            #             fit_result = self.get_fit(t_range,n_states_dict)

            #             fit_data[particle][n_state]['t'] = np.append(fit_data[particle][n_state]['t'], ti)
            #             fit_data[particle][n_state]['E0'] = np.append(fit_data[particle][n_state]['E0'], gv.mean(fit_result.p[particle+'_E0']))
            #             fit_data[particle][n_state]['E0_err'] = np.append(fit_data[particle][n_state]['E0_err'], gv.sdev(fit_result.p[particle+'_E0']))
            #             fit_data[particle][n_state]['chi2/df'] = np.append(fit_data[particle][n_state]['chi2/df'], fit_result.chi2 / fit_result.dof)
            #             fit_data[particle][n_state]['Q'] = np.append(fit_data[particle][n_state]['Q'], fit_result.Q)
            #             print(f"Appending data for particle {particle}, n_state {n_state}, time slice {t}")
            #             print(f"E0 value: {gv.mean(fit_result.p[particle+'_E0'])}, E0_err value: {gv.sdev(fit_result.p[particle+'_E0'])}")
            #             print("temp_fit.p[corr+'_E0']:", fit_result.p[particle+'_E0'])
            #             print("fit_data[n_state]['y']:", fit_data[particle][n_state]['E0'])

            #         # else:
            #         #     print(f"No data for n_state {n_state} at time slice {t}")  # New print statement

            #     cmap = matplotlib.cm.get_cmap('rainbow')
            #     min_max = lambda x : [np.min(x), np.max(x)]
            #     #minimum, maximum = min_max(fit_data['chi2/df'])
            #     norm = matplotlib.colors.Normalize(vmin=0.25, vmax=1.75)

            #     for i, n_state in enumerate(sorted(fit_data[particle].keys())):
            #         for j, ti in enumerate(fit_data[particle][n_state]['t']):
            #             color = cmap(norm(fit_data[particle][n_state]['chi2/df'][j]))
            #             y = gv.mean(fit_data[particle][n_state]['E0'][j])
            #             yerr = gv.sdev(fit_data[particle][n_state]['E0'][j])
            #             print(f"Processing particle {particle}, n_state {n_state}, time slice {ti}")
            #             print(f"y value: {y}, yerr value: {yerr}")

            #             alpha = 0.05
            #             plt.axvline(ti-0.5, linestyle='--', alpha=alpha)
            #             plt.axvline(ti+0.5, linestyle='--', alpha=alpha)

            #         # ti = ti + spacing[n_state-1]
            #         ti = ti 

            #         plt.errorbar(ti, y, xerr = 0.0, yerr=yerr, fmt=markers[n_state%len(markers)], mec='k', mfc='white', ms=10.0,
            #             capsize=5.0, capthick=2.0, elinewidth=5.0, alpha=0.9, ecolor=color, label=r"$N$=%s"%n_state)

            #     # best_fit = self.get_fit_forstab(t_range = [6,25],n_states={'hyperons':2})
            #     best_fit = self.get_fit_forstab()
            #     y_best = {}
            #     y_best[particle] = best_fit.p[particle+'_E0']
            #     ylabel = particle+'$E_0$'

            #     tp = np.arange(t[0]-1, t[-1]+2)
            #     tlim = (tp[0], tp[-1])

            #     pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
            #     y2 = np.repeat(pm(y_best, 0), len(tp))
            #     y2_upper = np.repeat(pm(y_best[particle], 1), len(tp))
            #     y2_lower = np.repeat(pm(y_best[particle], -1), len(tp))

            #     # Ground state plot
            #     plt.plot(tp, y2, '--')
            #     plt.plot(tp, y2_upper, tp, y2_lower)
            #     plt.fill_between(tp, y2_lower, y2_upper, facecolor = 'yellow', alpha = 0.25)

            #     plt.ylabel(ylabel, fontsize=24)
            #     plt.xlim(tlim[0], tlim[-1])

            #     # Limit y-axis when comparing multiple states
            #     if n_states_array is not None:
            #         plt.ylim(pm(y_best, -5), pm(y_best, 5))

            #     # Get unique markers when making legend
            #     handles, labels = plt.gca().get_legend_handles_labels()
            #     temp = {}
            #     for j, handle in enumerate(handles):
            #         temp[labels[j]] = handle

            #     plt.legend([temp[label] for label in sorted(temp.keys())], [label for label in sorted(temp.keys())])

            #     for i, particle in enumerate(sorted(fit_data.keys())):
            #         for j, n_state in enumerate(fit_data[particle]):
            #             print(f"Processing particle {particle}, n_state {n_state}")
            #             for k, ti in enumerate(fit_data[particle][n_state]['t']):
            #                 color = cmap(norm(fit_data[particle][n_state]['chi2/df'][k]))
            #                 y = gv.mean(fit_data[particle][n_state]['E0'][k])
            #                 yerr = gv.sdev(fit_data[particle][n_state]['E0_err'][k])

            #                 alpha = 0.05
            #                 plt.axvline(ti-0.5, linestyle='--', alpha=alpha)
            #                 plt.axvline(ti+0.5, linestyle='--', alpha=alpha)

            #                 ti = ti + spacing[j]
            #                 plt.errorbar(ti, y, xerr = 0.0, yerr=yerr, fmt=markers[i%len(markers)], mec='k', mfc='white', ms=10.0,
            #                     capsize=5.0, capthick=2.0, elinewidth=5.0, alpha=0.9, ecolor=color, label=r"$N$=%s"%n_state)

                

            #     t_middle = int((t_start + 2*t_end)/3)
            #     t  = np.arange(t_start, t_middle + 1)
            #     tp = np.arange(t[0]-1, t[-1]+2)
            #     tlim = (tp[0], tp[-1])

            #     pm = lambda x, k : gv.mean(x) + k*gv.sdev(x)
            #     y2 = np.repeat(pm(y_best[particle], 0), len(tp))
            #     y2_upper = np.repeat(pm(y_best[particle], 1), len(tp))
            #     y2_lower = np.repeat(pm(y_best[particle], -1), len(tp))

            #     # Ground state plot
            #     plt.plot(tp, y2, '--')
            #     plt.plot(tp, y2_upper, tp, y2_lower)
            #     plt.fill_between(tp, y2_lower, y2_upper, facecolor = 'yellow', alpha = 0.25)

            #     plt.ylabel(ylabel, fontsize=24)
            #     plt.xlim(tlim[0], tlim[-1])

            #     # Get unique markers when making legend
            #     handles, labels = plt.gca().get_legend_handles_labels()
            #     temp = {}
            #     for j, handle in enumerate(handles):
            #         temp[labels[j]] = handle

            #     plt.legend([temp[label] for label in sorted(temp.keys())], [label for label in sorted(temp.keys())])

            #     ###
            #     # Set axes: next for Q-values
            #     axQ = plt.axes([0.10,0.10,0.7,0.10])

            #     for i, particle in enumerate(sorted(fit_data.keys())):
            #         for j, n_state in enumerate(fit_data[particle]):
            #             for k, l in enumerate(fit_data[particle][n_state]['t']):
            #                 t = fit_data[particle][n_state]['t']
            #                 for ti in t:
            #                     alpha = 0.05
            #                     plt.axvline(ti-0.5, linestyle='--', alpha=alpha)
            #                     plt.axvline(ti+0.5, linestyle='--', alpha=alpha)
            #                 t = t
            #                 y = gv.mean(fit_data[particle][n_state]['Q'])
            #                 yerr = gv.sdev(fit_data[particle][n_state]['Q'])
            #                 color_data = fit_data[particle][n_state]['chi2/df']

            #                 sc = plt.scatter(t, y, vmin=0.25, vmax=1.75, marker=markers[i%len(markers)], c=color_data, cmap=cmap)

            #     # Set labels etc
            #     plt.ylabel('$Q$', fontsize=24)
            #     plt.xlabel('$t$', fontsize=24)
            #     plt.ylim(-0.05, 1.05)
            #     plt.xlim(tlim[0], tlim[-1])

            #     # Set axes: colorbar
            #     axC = plt.axes([0.85,0.10,0.05,0.80])

            #     colorbar = matplotlib.colorbar.ColorbarBase(axC, cmap=cmap,
            #                                 norm=norm, orientation='vertical')
            #     colorbar.set_label(r'$\chi^2_\nu$', fontsize = 24)

            #     fig = plt.gcf()
            #     plt.show()
            #     # else: plt.close()

            #     return fig

                    # # Plot E0 fit parameters with error bars
                    # axs[0].errorbar(t_values, E0_values, yerr=E0_errors, fmt=markers[n_state], color=colors[particle_idx], label=f'n_state = {n_state}')

                    # axs[0].set_title('Hyperon stability plot for n=2 states')
                    # axs[0].set_ylabel('E0 fit parameter')
                    # axs[0].legend()

                    # # Q-value plot
                    # ax = axs[1]
                    # ax.plot(t_values, q_values, marker='o', linestyle='')
                    # ax.set_xlabel('Starting time slice')
                    # ax.set_ylabel('Q-value')

                    # # Chi2/dof color map
                    # cmap = matplotlib.cm.get_cmap('rainbow')
                    # norm = matplotlib.colors.Normalize(vmin=0.25, vmax=1.75)
                    # # chi2_dof_values = [ffit_result[n_state].chi2 / [fit_result[n_state].dof for t, [fit_result[n_state] in all_hyperons.items()]
                    # colors = [cmap(norm(chi2_dof)) for chi2_dof in chi2_values]

                    # for t, chi2_dof, color in zip(t_values, chi2_values, colors):
                    #     rect = plt.Rectangle((t - 0.5, 0), 1, chi2_dof, color=color)
                    #     axs[1].add_patch(rect)

                    # cax = fig.add_axes([0.95, 0.15, 0.02,0.7])
                    # colorbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
                    # colorbar.set_label(r'$\chi^2_\nu$', fontsize=24) 
                    # pdf.savefig(fig, bbox_inches='tight') 
                    # plt.close(fig)  
                    # plt.show()