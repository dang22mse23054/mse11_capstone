// /* global XMLHttpRequest */
import { Popper, TextField, withStyles, createStyles } from '@material-ui/core';
import Autocomplete, { createFilterOptions } from '@material-ui/lab/Autocomplete';
import { Status } from 'constDir';
import {
	IBaseAutoCompleteOption as IOptionItem, IGraphqlEdge, IGraphqlPageInfo, IGraphqlPagingObj
} from 'interfaceDir';
import React, { ReactNode } from 'react';
import { ApiRequest } from 'servDir';

interface IProps {
	id?: string
	label?: string
	isError?: boolean
	isShrink?: boolean
	isMultiple?: boolean
	placeholder?: string
	variant?: 'filled' | 'outlined' | 'standard'
	size?: 'medium' | 'small'
	options?: Array<any>,
	disabled?: boolean
	typeInput?: string
	autoInit?: boolean
	searchOnEmptyKeyword: boolean
	value?: Array<any>
	defaultValue?: Array<any>
	remoteUrl?: string
	limitTags?: number
	clearOnBlur?: boolean
	clearable?: boolean
	freeSolo?: boolean
	open?: boolean
	noOptionsText?: string
	loadingText?: string
	noDataText?: string
	helperText?: string
	closeOnSelect?: boolean
	blurOnSelect?: boolean
	createOnBlur?: boolean
	isSingleMode?: boolean
	gqlMethod: string
	queryString: string
	componentName: string
	isCAUser: boolean
	getFilterStr?(option): any
	onChange?(newChips, reason): any
	getOptionLabel?(option: IOptionItem): string
	renderOption?(option: IOptionItem): any
	renderTags?(values: Array<IOptionItem>, getTagProps): any
	groupBy?(option: IOptionItem): string
	renderGroup?(option: IOptionItem): ReactNode
	renderInput?(params: any): ReactNode
	onInputChange?(event: any, keyword: string, reason: string): void
	getOptionSelected?(optionItem: IOptionItem, selectedItem: IOptionItem): boolean
	filterOptions?(option: IOptionItem, state: any): Array<IOptionItem>
	parseGqlData?(edge: IGraphqlEdge): IOptionItem
	convertOptions?(options: Array<any>, mode: 'new' | 'append'): Array<IOptionItem>
	searchVariables?(keyword, nextCursor): any
	searchCallback?(value?: any): any
}

export interface IState extends IGraphqlPageInfo {
	suggestions?: Array<any>,
	chips?: Array<any>,
	deletedChips: Array<any>
	textField?: string,
	isRemoteLoading: boolean
	currentPage: number
	totalPage: number
	searchKeyword: string
	message: string
}

const useStyles = (theme: Theme) => createStyles({
	tag: {
		// maxWidth: '80%'
	},
	input: {
		minWidth: '0 !important',
		paddingInline: '0 !important',
	},
	focused: {
		'& input': {
			minWidth: '30px !important'
		}
	},
});


const CustomPopper = (props) => (
	<Popper {...props} placement='bottom-start' style={{ minWidth: props.style.width, maxWidth: 500 }} />
);

class ReactAutosuggest extends React.Component<IProps, IState> {
	private timer = -1;
	private keyword = '';

	static defaultProps: IProps = {
		onChange: (newChips, reason) => false,
		getFilterStr: (option) => {
			return option.key;
		},
		searchVariables: (keyword, nextCursor) => ({ keyword, nextCursor }),
		searchCallback: () => true,
		id: 'default_id',
		label: 'Title',
		isError: false,
		isMultiple: true,
		disabled: false,
		autoInit: false,
		searchOnEmptyKeyword: false,
		placeholder: '',
		helperText: '',
		variant: 'standard',
		size: 'medium',
		value: [],
		defaultValue: [],
		limitTags: -1,
		clearOnBlur: true,
		freeSolo: false,
		noOptionsText: 'データが見つかりませんでした',
		closeOnSelect: false,
		createOnBlur: false,
		isSingleMode: false,
		componentName: 'ReactAutosuggest',
		isCAUser: false
	}

	state = {
		suggestions: this.props.options || [],
		chips: [],
		deletedChips: [],
		textField: '',
		isRemoteLoading: false,
		currentPage: 0,
		totalPage: 0,
		searchKeyword: '',
		message: this.props.loadingText || '検索中...'
	};

	componentDidMount = () => {
		this.initValue(this.props.value || this.props.defaultValue);
		if (this.props.autoInit) {
			this.searchOnServer('', this.props.searchCallback);
		}
	}

	componentDidUpdate = (prevProps: IProps, prevState: IState, snapshot) => {
		if (prevProps.value != this.props.value) {
			this.initValue(this.props.value);
		}
		if (prevProps.options != this.props.options) {
			this.setState({ suggestions: this.props.options });
		}

		// if (this.props.autoInit && this.state.searchKeyword == '' && prevState.searchKeyword != this.state.searchKeyword) {
		// 	console.log('search.....')
		// 	this.searchOnServer('');
		// }

	}

	initValue = (value) => {
		// convert value to chips and deletedChips
		const chips: Array<any> = [];
		const deletedChips: Array<any> = [];
		if (value) {
			value.map((chip, index) => {
				let item = chip;

				if (item && !item.key && !item.value) {
					item = {
						key: `${chip.id}`,
						value: { ...chip }
					};
				}
				if (item.value.status == Status.DELETED) {
					deletedChips.push(item);
				} else {
					chips.push(item);
				}
			});
		}

		this.setState({
			chips,
			deletedChips
		});
	}

	clearTimer = () => {
		if (this.timer) {
			clearTimeout(this.timer);
		}
	}

	handleClose = () => {
		this.setState({
			suggestions: this.state.searchKeyword != '' ? [] : this.state.suggestions,
			isRemoteLoading: false
		});

		// add chip on Blur (for TextInputTag only)
		if (this.props.createOnBlur && !this.props.remoteUrl && this.keyword) {
			const reason = 'create-option';
			const chip = {
				key: this.keyword,
				value: {
					id: null,
					value: this.keyword,
					order: this.state.chips.length + 1,
					status: Status.NEW
				}
			};
			this.handleAddChip(reason, chip);
		}
		this.keyword = '';

		if (this.props.searchOnEmptyKeyword && this.state.searchKeyword != this.keyword) {
			this.searchOnServer('', this.props.searchCallback);
		}
	}

	searchOnServer = (keyword: string, callback: (value?: any) => any) => {
		Promise.resolve()
			.then(async (res) => {
				return await ApiRequest.sendPOST(`${this.props.remoteUrl}`, {
					operationName: this.props.gqlMethod,
					query: this.props.queryString,
					variables: this.props.searchVariables(keyword)
				}, async (response) => {
					const obj: IGraphqlPagingObj<any> = response.data.data[this.props.gqlMethod];

					let newState = null;
					let suggestions = obj.edges.map(this.props.parseGqlData);
					if (obj.edges.length == 0) {
						newState = {
							suggestions,
							hasNext: false,
							next: null,
							searchKeyword: '',
							message: (this.props.noDataText || 'データが見つかりませんでした'),
						};
					} else {

						if (this.props.convertOptions) {
							suggestions = await this.props.convertOptions(suggestions, 'new');
						}

						newState = {
							suggestions,
							...obj.pageInfo,
							searchKeyword: keyword,
							message: this.props.loadingText || '検索中...',
						};
					}

					this.setState(newState);
					return suggestions;

				}, (error) => {
					console.error(error);
					this.setState({ message: 'データを取得できませんでした' });
				});
			})
			.then((suggestions) => {
				this.setState({ isRemoteLoading: false });
				if (suggestions.length && this.props.isCAUser) {
				
					this.timer = window.setTimeout(() => {
						if ( keyword.includes(',')) {
							!this.props.isSingleMode ? suggestions.forEach((suggestion: any) => this.handleAddChip('select-option', suggestion)) : this.handleChangeChip('select-option', suggestions);
						}
						
					}, 1000);
				}
			})
			.then(callback)
			.catch((e) => console.error(e));
	}

	handleOnScroll = () => {
		if (!this.state.isRemoteLoading && this.state.next) {
			this.setState({ isRemoteLoading: true });
			Promise.resolve()
				.then(async (res) => {
					return await ApiRequest.sendPOST(this.props.remoteUrl, {
						operationName: this.props.gqlMethod,
						query: this.props.queryString,
						variables: this.props.searchVariables(this.state.searchKeyword, this.state.next)

					}, async (response) => {
						const obj: IGraphqlPagingObj<ICAUser> = response.data.data[this.props.gqlMethod];


						let newState = null;
						if (obj.edges.length == 0) {
							newState = {
								hasNext: false,
								next: null,
							};
						} else {

							let suggestions = obj.edges.map(this.props.parseGqlData);

							if (this.props.convertOptions) {
								suggestions = await this.props.convertOptions(suggestions, 'append');
							}

							newState = {
								suggestions: this.state.suggestions.concat(suggestions),
								...obj.pageInfo,
							};
						}

						this.setState(newState);

					}, (error) => {
						console.error(error);
					});
				})
				.then(() => this.setState({ isRemoteLoading: false }))
				.catch((e) => console.error(e));
		}
	}

	handleAddChip = (reason, chip) => {
		// Stop request to remote server
		this.clearTimer();
		this.setState({ isRemoteLoading: false });

		const allChips = [...this.state.chips].concat(this.state.deletedChips);

		let oldChip = null;
		let oldChipIndex = -1;

		let alreadyExist = false;
		let alreadyDeleted = false;
		const activeChipMaxIndex = this.state.chips.length - 1;

		// search all to get the chip that the same key
		allChips.map((alreadyChip, index) => {
			if (alreadyChip.key === chip.key) {
				alreadyExist = true;
				if (index > activeChipMaxIndex) {
					// specify that chip has been deleted (search in deletedChips)
					alreadyDeleted = true;
					oldChip = alreadyChip;
					// get index of this item in deletedChips list
					oldChipIndex = index - this.state.chips.length;

				}
			}
		});

		// only accept to add the selected item to chips if currentChip = null OR currentChip.status = deleted
		if (!alreadyExist) {
			// Not have this chip before => create new
			const newChips = this.state.chips.concat([chip]);
			// Update to thix component
			this.setState({ chips: newChips });
			// run callback
			if (this.props.onChange) { this.props.onChange(newChips.concat(this.state.deletedChips), reason); }

		} else if (alreadyDeleted) {
			// Having a chip before 
			// remove this chip from deletedChips
			const delChips = [...this.state.deletedChips];
			delChips.splice(oldChipIndex, 1);

			// add 'renewed' chip
			const newChips = [...this.state.chips];
			newChips.push({
				...oldChip,
				value: {
					...oldChip.value,
					status: Status.UPDATED
				}

			});

			// update state again
			this.setState({
				chips: newChips,
				deletedChips: delChips
			});
			// run callback
			if (this.props.onChange) { this.props.onChange(newChips.concat(delChips), reason); }
		}
	}

	handleChangeChip = (reason, chip) => {
		// Stop request to remote server
		this.clearTimer();
		this.setState({ isRemoteLoading: false });

		const allChips = [...this.state.chips].concat(this.state.deletedChips);

		let oldChip = null;
		let oldChipIndex = -1;

		let alreadyDeleted = false;
		const activeChipMaxIndex = this.state.chips.length - 1;

		// search all to get the chip that the same key
		allChips.map((alreadyChip, index) => {
			if (alreadyChip.key === chip.key) {
				if (index > activeChipMaxIndex) {
					// specify that chip has been deleted (search in deletedChips)
					alreadyDeleted = true;
					oldChip = alreadyChip;
					// get index of this item in deletedChips list
					oldChipIndex = index - this.state.chips.length;

				}
			}
		});
		const newChip = [];
		const delChips = [...this.state.deletedChips];
		if (alreadyDeleted) {
			// Having a chip before 
			// remove this chip from deletedChips
			delChips.splice(oldChipIndex, 1);

			newChip.push({
				...oldChip,
				value: {
					...oldChip.value,
					status: Status.UPDATED
				}

			});
		} else {
			newChip.push(chip);
		}

		const delChip = this.state.chips[0];
		if (delChip) {
			delChips.push({
				...delChip,
				value: {
					...delChip.value,
					status: Status.DELETED
				}
			});
		}

		// update state again
		this.setState({
			chips: newChip,
			deletedChips: delChips
		});

		if (this.props.onChange) {
			this.props.onChange(newChip.concat(delChips), reason);
		}
	}

	handleDeleteChip = (afterChangedList: Array<any>, reason, detail) => {
		const chip = detail.option;
		const newState = { chips: [], deletedChips: [] };

		// If this chip has ID => already in DB => need to update in the future
		if (chip.value.id != null) {
			newState.deletedChips = this.state.deletedChips.concat([{
				...chip,
				value: {
					...chip.value,
					status: Status.DELETED
				}
			}]);
		}

		newState.chips = afterChangedList;

		// Update to thix component
		this.setState(newState);

		// run callback
		if (this.props.onChange) { this.props.onChange(afterChangedList.concat(newState.deletedChips), reason); }
	}

	handleClearAll = (afterChangedList: Array<any>, reason) => {
		const deletedChips = this.state.chips.map((chip, index) => {
			// clone new Object to set status
			return {
				...chip,
				value: {
					...chip.value,
					status: Status.DELETED
				}
			};
		});

		// Update to thix component
		this.setState({
			chips: afterChangedList,
			deletedChips

		});
		// run callback
		if (this.props.onChange) { this.props.onChange(afterChangedList.concat(deletedChips), reason); }
	}

	render() {
		const {
			classes, renderTags, renderInput, renderOption, renderGroup, helperText,
			groupBy, open, filterOptions, disabled,
			getOptionLabel, getOptionSelected, onInputChange,
		} = this.props;

		return (
			<Autocomplete
				classes={{
					tag: classes.tag,
					input: classes.input,
					focused: classes.focused,
				}}
				style={{ display: 'flex' }}
				ListboxProps={{
					onScroll: (event: React.SyntheticEvent) => {
						const listboxNode = event.currentTarget;
						if (listboxNode.scrollTop + listboxNode.clientHeight >= listboxNode.scrollHeight - 20) {
							this.handleOnScroll();
						}
					},
					style: { padding: 0 }
				}}
				autoHighlight
				disabled={disabled}
				inputMode='search'
				filterSelectedOptions
				multiple={this.props.isMultiple}
				disableCloseOnSelect={!this.props.closeOnSelect}
				blurOnSelect={this.props.blurOnSelect}
				disableClearable={!this.props.clearable}
				limitTags={this.props.limitTags}
				id={this.props.id}
				size={this.props.size}
				clearOnBlur={this.props.clearOnBlur}
				freeSolo={this.props.freeSolo}
				loading={this.state.isRemoteLoading}
				noOptionsText={this.props.noOptionsText}
				clearText={'クリア'}
				loadingText={this.state.message}

				PopperComponent={CustomPopper}

				onClose={this.handleClose}
				onChange={(event, afterChangedList, reason, detail) => {
					switch (reason) {

						// add chip on Enter (freeSolo=true only)
						case 'create-option':
							this.clearTimer();
							this.setState({
								suggestions: [],
								isRemoteLoading: true
							});
							const chip = {
								key: detail.option,
								value: {
									id: null,
									value: detail.option,
									order: this.state.chips.length + 1,
									status: Status.NEW
								}
							};

							!this.props.isSingleMode ? this.handleAddChip(reason, chip) : this.handleChangeChip(reason, chip);
							break;

						case 'select-option':
							!this.props.isSingleMode ? this.handleAddChip(reason, detail.option) : this.handleChangeChip(reason, detail.option);
							break;

						case 'remove-option':
							this.handleDeleteChip(afterChangedList, reason, detail);
							break;

						case 'clear':
							this.handleClearAll(afterChangedList, reason);
							break;
					}
				}}

				// For debug
				{...(open ? { open } : {})}

				{...(renderTags ? { renderTags } : {})}
				{...(groupBy ? { groupBy } : {})}
				{...(renderGroup ? { renderGroup } : {})}
				renderInput={renderInput || (
					(params) => {
						return <TextField
							error={this.props.isError}
							{...params}
							{...(helperText && { helperText })}
							type={this.props.typeInput}
							InputLabelProps={{ shrink: this.props.isShrink }}
							variant={this.props.variant}
							label={this.props.label}
							placeholder={this.state.chips.length == 0 ? this.props.placeholder : ''}
						/>;
					}
				)}
				onInputChange={onInputChange || (
					(event, keyword, reason) => {
						switch (reason) {
							case 'input':
								this.keyword = keyword;
								if (this.props.remoteUrl) {
									this.clearTimer();
									// hide Loading message
									this.setState({ isRemoteLoading: false });
									if (this.props.searchOnEmptyKeyword || keyword != '') {
										// show Loading message and wait for sending request 
										this.setState({
											isRemoteLoading: true,
											message: '検索中...'
										});
										this.timer = window.setTimeout(() => {
											this.searchOnServer(keyword, this.props.searchCallback);
										}, 1000);
									}
								}
								break;

							case 'reset':
							case 'clear':
								break;
						}
					}
				)}
				filterOptions={filterOptions || (
					createFilterOptions({
						// matchFrom: 'start',
						stringify: this.props.getFilterStr
					})
				)}

				getOptionLabel={getOptionLabel || (
					(option) => option.key
				)}

				getOptionSelected={getOptionSelected || (
					(optionItem, selectedItem) => {
						// this function is using for consider if the selectedItem is the same with optionItem
						// If true (the same together) => NOT show in the suggestion
						// console.log(optionItem.key === selectedItem.key && optionItem.value.status === selectedItem.value.status && selectedItem.value.status === Status.DELETED)
						return optionItem.key === selectedItem.key;
					}
				)}

				{...(renderOption ? { renderOption } : {})}
				options={this.state.suggestions}
				value={this.state.chips}
			/>
		);
	}
}

export default withStyles(useStyles)(ReactAutosuggest);
export { IProps, IState };
