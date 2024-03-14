import React, { forwardRef, Fragment } from 'react';
import ColorButton, { IProps as IColorBtnProps } from 'compDir/Button';
import Menu from '@material-ui/core/Menu';
import MenuItem from '@material-ui/core/MenuItem';
import CustomLabel from 'compDir/CustomLabel';

export interface MenuOption {
	label: string
	icon?: any
	value?: any
	disabled?: boolean
	onClick?(): any
}

interface IProps extends IColorBtnProps {
	menuOptions: Array<MenuOption | string>
	closeAfterClick?: boolean
	onSelect?(): any
}

const defaultProps: IProps = {
	menuOptions: [],
	closeAfterClick: true
};

const MenuButton: React.FC<IProps> = forwardRef((props, ref) => {
	const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

	const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
		setAnchorEl(event.currentTarget);
	};

	const handleClose = () => {
		setAnchorEl(null);
	};

	const handleMenuClick = (value, callback, closeAfterClick) => {
		if (callback && typeof callback === 'function') {
			callback(value);
		}
		if (closeAfterClick) {
			handleClose();
		}
	};

	const { menuOptions, closeAfterClick, onSelect, ...restProps } = props;

	return (
		<Fragment>
			<ColorButton {...restProps} ref={ref} aria-controls="simple-menu" aria-haspopup="true" onClick={handleClick}>
				{props.children}
			</ColorButton>
			<Menu id="simple-menu" anchorEl={anchorEl} keepMounted
				getContentAnchorEl={undefined}
				anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
				transformOrigin={{ vertical: 'top', horizontal: 'center' }}
				open={Boolean(anchorEl)} onClose={handleClose}>
				{
					menuOptions.map((opt: MenuOption | string, index: number) => {
						let callback = onSelect;
						let label = opt;
						let value = label;
						let icon = null;
						let disabled = false;

						if (typeof opt !== 'string') {
							callback = (opt.onClick || onSelect);
							label = opt.label;
							value = opt.value || label;
							icon = opt.icon;
							disabled = Boolean(opt.disabled);
						}

						return (
							<MenuItem key={index} disabled={disabled} onClick={() => handleMenuClick(value, callback, closeAfterClick || true)}>
								{
									icon ? <CustomLabel value={label} icon={icon} spacing={2} /> : label
								}
							</MenuItem>
						);
					})
				}
			</Menu>

		</Fragment>
	);
});

MenuButton.defaultProps = defaultProps;

export default MenuButton;