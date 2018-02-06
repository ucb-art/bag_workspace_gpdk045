# -*- coding: utf-8 -*-

import os
import pkg_resources
from typing import Dict

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'tb_mos_ft.yaml'))


# noinspection PyPep8Naming
class ee240b_mos_char__tb_mos_ft(Module):
    """Module for library ee240b_mos_char cell tb_mos_ft.

    Fill in high level description here.
    """

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            mos_type='foo',
            lch='bar',
            threshold='baz',
        )

    def design(self, mos_type, lch, threshold):
        """To be overridden by subclasses to design this module.

        This method should fill in values for all parameters in
        self.parameters.  To design instances of this module, you can
        call their design() method or any other ways you coded.

        To modify schematic structure, call:

        rename_pin()
        delete_instance()
        replace_instance_master()
        reconnect_instance_terminal()
        restore_instance()
        array_instance()
        """
        if mos_type == 'pch':
            self.replace_instance_master('XDUT', 'BAG_prim', 'pmos4_standard')
        self.instances['XDUT'].design(w=0.5e-6, l=lch, nf=20, intent=threshold)
